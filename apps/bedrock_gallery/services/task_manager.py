import asyncio
from typing import Dict, Optional
from datetime import datetime
import streamlit as st
from apps.bedrock_gallery.types import MediaType
from services.bedrock_service import get_video_job
from services.storage_service import StorageService


class AsyncTaskManager:
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.tasks = {}
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._restore_pending_tasks()
        
    def start_video_polling(self, task_id: str, invocation_arn: str):
        self.tasks[task_id] = {
            'status': 'InProgress',
            'invocation_arn': invocation_arn,
            'last_checked': datetime.now()
        }
        asyncio.run_coroutine_threadsafe(
            self._poll_video_status(task_id), 
            self.loop
        )

    async def _poll_video_status(self, task_id: str):
        while task_id in self.tasks and self.tasks[task_id]['status'] not in ['Completed', 'Failed']:
            try:
                invocation = get_video_job(self.tasks[task_id]['invocation_arn'])
                status = invocation.get('status', 'InProgress')
                print(f"Polling Job: {invocation}, {status}")
                
                if status != self.tasks[task_id]['status']:
                    self.tasks[task_id].update({
                        'status': status,
                        'last_checked': datetime.now(),
                        'metadata': self.storage_service.update_video_status(
                            id=task_id,
                            model_type=invocation.get('modelType', ''),
                            prompt=invocation.get('prompt', ''),
                            details=invocation
                        )
                    })
                
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"Error polling video status: {str(e)}")
                self.tasks[task_id]['status'] = 'Failed'
                break
        
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)
    
    def _restore_pending_tasks(self):
        pending_tasks = self.storage_service.get_media_list(media_type=MediaType.VIDEO.value)
        for task in pending_tasks:
            if task.get('details', {}).get('status') == 'InProgress':
                self.start_video_polling(
                    task_id=task['id'],
                    invocation_arn=task['details']['invocationArn']
                )