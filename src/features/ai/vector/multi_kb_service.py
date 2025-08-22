from typing import List, Dict, Any, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ..vector.vector_store import VectorStoreService
from ..utils.conversation_utils import format_conversation_context

logger = logging.getLogger(__name__)

class MultiKnowledgeBaseService:
    """Service for handling multi-knowledge base operations with performance optimizations."""
    
    def __init__(self, vector_config: Dict[str, Any]):
        self.vector_service = VectorStoreService(config=vector_config)
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="multi_kb")
    
    async def query_multiple_knowledge_bases(
        self,
        user_id: str,
        query: str,
        knowledge_base_ids: List[str],  # Fixed parameter name
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        primary_kb_id: Optional[str] = None  # Optional primary KB
    ) -> List[Dict[str, Any]]:
        """Query multiple knowledge bases in parallel and merge results."""
        
        if not knowledge_base_ids:
            logger.warning("No knowledge base IDs provided")
            return []
        
        # Separate primary and additional KBs
        if primary_kb_id and primary_kb_id in knowledge_base_ids:
            additional_kb_ids = [kb_id for kb_id in knowledge_base_ids if kb_id != primary_kb_id]
        else:
            primary_kb_id = knowledge_base_ids[0]
            additional_kb_ids = knowledge_base_ids[1:]
        
        # Create parallel tasks for all knowledge bases
        tasks = []
        kb_task_mapping = {}
        
        # Primary KB task (higher k value)
        primary_task = asyncio.create_task(
            self._query_single_kb(
                user_id=user_id,
                query=query,
                kb_id=primary_kb_id,
                k=k,
                metadata_filter=metadata_filter,
                source_type='primary'
            )
        )
        tasks.append(primary_task)
        kb_task_mapping[primary_task] = primary_kb_id
        
        # Additional KB tasks (lower k value for efficiency)
        additional_k = max(1, k // 2) if additional_kb_ids else 0
        for kb_id in additional_kb_ids:
            task = asyncio.create_task(
                self._query_single_kb(
                    user_id=user_id,
                    query=query,
                    kb_id=kb_id,
                    k=additional_k,
                    metadata_filter=metadata_filter,
                    source_type='additional'
                )
            )
            tasks.append(task)
            kb_task_mapping[task] = kb_id
        
        # Execute all queries in parallel
        logger.info(f"Executing {len(tasks)} parallel KB queries")
        start_time = asyncio.get_event_loop().time()
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in parallel KB queries: {e}")
            return []
        
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Parallel KB queries completed in {execution_time:.3f}s")
        
        # Process results
        all_results = []
        kb_sources = {}
        
        for i, result in enumerate(results):
            task = tasks[i]
            kb_id = kb_task_mapping[task]
            
            if isinstance(result, Exception):
                logger.warning(f"Failed to query KB {kb_id}: {result}")
                continue
            
            if result:
                all_results.extend(result)
                kb_sources[kb_id] = len(result)
        
        # Sort by relevance score and date
        all_results.sort(key=lambda x: (
            -x.get('score', 0),  # Higher score first (negative for descending)
            x.get('dateAdded', '')  # Then by date (ascending)
        ))
        
        # Limit total results with smart distribution
        final_results = self._smart_result_distribution(all_results, k * 2, kb_sources)
        
        logger.info(f"Multi-KB query returned {len(final_results)} results from {len(kb_sources)} knowledge bases")
        return final_results
    
    async def _query_single_kb(
        self,
        user_id: str,
        query: str,
        kb_id: str,
        k: int,
        metadata_filter: Optional[Dict[str, Any]],
        source_type: str
    ) -> List[Dict[str, Any]]:
        """Query a single knowledge base with error handling."""
        try:
            results = await self.vector_service.query_chat_history(
                userId=user_id,
                query=query,
                knowledgebaseId=kb_id,
                k=k,
                metadata_filter=metadata_filter
            )
            
            # Tag results with source information
            for result in results:
                result['_source_kb'] = kb_id
                result['_source_type'] = source_type
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to query KB {kb_id}: {e}")
            return []
    
    def _smart_result_distribution(
        self, 
        all_results: List[Dict[str, Any]], 
        max_results: int,
        kb_sources: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Distribute results smartly across knowledge bases."""
        if len(all_results) <= max_results:
            return all_results
        
        # Ensure representation from each KB
        final_results = []
        kb_counts = {kb_id: 0 for kb_id in kb_sources.keys()}
        min_per_kb = max(1, max_results // len(kb_sources))
        
        # First pass: ensure minimum representation
        for result in all_results:
            kb_id = result.get('_source_kb')
            if kb_id and kb_counts[kb_id] < min_per_kb:
                final_results.append(result)
                kb_counts[kb_id] += 1
                
                if len(final_results) >= max_results:
                    break
        
        # Second pass: fill remaining slots with best results
        if len(final_results) < max_results:
            remaining_results = [r for r in all_results if r not in final_results]
            remaining_slots = max_results - len(final_results)
            final_results.extend(remaining_results[:remaining_slots])
        
        return final_results
    
    async def get_conversation_from_multiple_kbs(
        self,
        user_id: str,
        conversation_id: str,
        primary_kb_id: str,
        additional_kb_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get conversation messages from multiple knowledge bases in parallel."""
        
        # Create parallel tasks
        tasks = []
        kb_task_mapping = {}
        
        # Primary KB task
        primary_task = asyncio.create_task(
            self._get_conversation_single_kb(user_id, conversation_id, primary_kb_id, 'primary')
        )
        tasks.append(primary_task)
        kb_task_mapping[primary_task] = primary_kb_id
        
        # Additional KB tasks
        for kb_id in additional_kb_ids:
            if kb_id != primary_kb_id:
                task = asyncio.create_task(
                    self._get_conversation_single_kb(user_id, conversation_id, kb_id, 'additional')
                )
                tasks.append(task)
                kb_task_mapping[task] = kb_id
        
        # Execute in parallel
        logger.info(f"Executing {len(tasks)} parallel conversation queries")
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in parallel conversation queries: {e}")
            return []
        
        # Process results
        all_messages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                kb_id = kb_task_mapping[task]
                logger.warning(f"Failed to get conversation from KB {kb_id}: {result}")
                continue
            
            if result:
                all_messages.extend(result)
        
        # Remove duplicates and sort
        unique_messages = self._deduplicate_messages(all_messages)
        unique_messages.sort(key=lambda x: x.get('dateAdded', ''))
        
        logger.info(f"Retrieved {len(unique_messages)} unique messages from {len(tasks)} knowledge bases")
        return unique_messages
    
    async def _get_conversation_single_kb(
        self,
        user_id: str,
        conversation_id: str,
        kb_id: str,
        source_type: str
    ) -> List[Dict[str, Any]]:
        """Get conversation from a single KB with error handling."""
        try:
            messages = await self.vector_service.get_conversation_by_id(
                userId=user_id,
                conversationId=conversation_id,
                knowledgebaseId=kb_id
            )
            
            # Tag messages with source
            for msg in messages:
                msg['_source_kb'] = kb_id
                msg['_source_type'] = source_type
            
            return messages
            
        except Exception as e:
            logger.warning(f"Failed to get conversation from KB {kb_id}: {e}")
            return []
    
    def _deduplicate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate messages efficiently."""
        seen_messages = set()
        unique_messages = []
        
        for msg in messages:
            # Create a unique key based on ID, content, and timestamp
            key = (
                msg.get('id', ''),
                msg.get('body', '')[:100],  # First 100 chars to handle minor variations
                msg.get('dateAdded', '')
            )
            
            if key not in seen_messages:
                seen_messages.add(key)
                unique_messages.append(msg)
        
        return unique_messages
    
    def merge_contexts(self, results: List[Dict[str, Any]]) -> str:
        """Merge results from multiple knowledge bases into a single context efficiently."""
        
        if not results:
            return ""
        
        # Group by source KB efficiently
        kb_groups = {}
        for result in results:
            kb_id = result.get('_source_kb', 'unknown')
            if kb_id not in kb_groups:
                kb_groups[kb_id] = []
            kb_groups[kb_id].append(result)
        
        # Build context with streaming approach
        context_parts = []
        
        # Process primary KB first
        for kb_id, kb_results in kb_groups.items():
            source_type = kb_results[0].get('_source_type', 'unknown')
            
            # Only format if we have meaningful content
            if kb_results:
                kb_context = format_conversation_context(kb_results)
                if kb_context and kb_context.strip():
                    priority_marker = "[PRIMARY] " if source_type == 'primary' else "[ADDITIONAL] "
                    context_parts.append(f"{priority_marker}=== Knowledge Base: {kb_id} ({source_type}) ===\n{kb_context}")
        
        return "\n\n".join(context_parts)
    
    async def is_multi_kb_query(self, additional_kb_ids: List[str]) -> bool:
        """Check if this is a multi-knowledge base query."""
        return bool(additional_kb_ids and len(additional_kb_ids) > 0)
    
    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)