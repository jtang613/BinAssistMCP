"""
Async task management for BinAssistMCP

This module provides task management for long-running operations like
decompilation and analysis, preventing blocking of the MCP connection.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import threading

from .logging import log


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class McpTask:
    """Represents an async MCP task"""
    id: str
    name: str
    status: TaskStatus
    progress: float = 0.0  # 0.0 to 1.0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


class TaskManager:
    """Manages async tasks for long-running operations"""

    def __init__(self, max_tasks: int = 100, cleanup_completed_after: int = 300):
        """Initialize the task manager.

        Args:
            max_tasks: Maximum number of tasks to keep in memory
            cleanup_completed_after: Seconds after which completed tasks are removed
        """
        self._tasks: Dict[str, McpTask] = {}
        self._lock = threading.RLock()
        self._max_tasks = max_tasks
        self._cleanup_after = cleanup_completed_after
        self._running_futures: Dict[str, asyncio.Future] = {}

    async def submit(self, func: Callable, name: str = "task", *args, **kwargs) -> str:
        """Submit a task for async execution.

        Args:
            func: Callable to execute (can be sync or async)
            name: Human-readable task name
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())

        with self._lock:
            # Cleanup old tasks if needed
            self._cleanup_old_tasks()

            task = McpTask(
                id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                progress=0.0
            )
            self._tasks[task_id] = task

        # Start the task execution
        asyncio.create_task(self._run_task(task_id, func, *args, **kwargs))
        log.log_info(f"Task {task_id} ({name}) submitted")

        return task_id

    async def _run_task(self, task_id: str, func: Callable, *args, **kwargs):
        """Internal method to run a task."""
        with self._lock:
            if task_id not in self._tasks:
                return
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))

            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.status = TaskStatus.COMPLETED
                    task.progress = 1.0
                    task.result = result
                    task.completed_at = datetime.now()

            log.log_info(f"Task {task_id} completed successfully")

        except asyncio.CancelledError:
            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
            log.log_info(f"Task {task_id} was cancelled")

        except Exception as e:
            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
            log.log_error(f"Task {task_id} failed: {e}")

    def get_task(self, task_id: str) -> Optional[McpTask]:
        """Get a task by ID.

        Args:
            task_id: Task ID to retrieve

        Returns:
            McpTask if found, None otherwise
        """
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status as a dictionary.

        Args:
            task_id: Task ID to check

        Returns:
            Task status dictionary
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {
                    "id": task_id,
                    "error": "Task not found",
                    "status": "unknown"
                }
            return task.to_dict()

    def update_progress(self, task_id: str, progress: float, metadata: Dict[str, Any] = None):
        """Update task progress.

        Args:
            task_id: Task ID to update
            progress: Progress value (0.0 to 1.0)
            metadata: Optional metadata to update
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.progress = max(0.0, min(1.0, progress))
                if metadata:
                    task.metadata.update(metadata)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancellation was initiated, False otherwise
        """
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

        # Cancel the asyncio task if running
        if task_id in self._running_futures:
            future = self._running_futures.get(task_id)
            if future and not future.done():
                future.cancel()

        log.log_info(f"Task {task_id} cancellation requested")
        return True

    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """List all tasks, optionally filtered by status.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of task dictionaries
        """
        with self._lock:
            tasks = []
            for task in self._tasks.values():
                if status_filter is None or task.status == status_filter:
                    tasks.append(task.to_dict())
            return tasks

    def _cleanup_old_tasks(self):
        """Clean up old completed tasks to prevent memory growth."""
        now = datetime.now()
        tasks_to_remove = []

        for task_id, task in self._tasks.items():
            # Remove completed tasks older than cleanup_after
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.completed_at:
                    age = (now - task.completed_at).total_seconds()
                    if age > self._cleanup_after:
                        tasks_to_remove.append(task_id)

        # Also remove excess tasks if we're over the limit
        if len(self._tasks) - len(tasks_to_remove) >= self._max_tasks:
            # Remove oldest completed tasks first
            completed_tasks = [
                (tid, t) for tid, t in self._tasks.items()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                and tid not in tasks_to_remove
            ]
            completed_tasks.sort(key=lambda x: x[1].completed_at or datetime.min)

            excess = len(self._tasks) - len(tasks_to_remove) - self._max_tasks + 10
            for tid, _ in completed_tasks[:excess]:
                tasks_to_remove.append(tid)

        for task_id in tasks_to_remove:
            del self._tasks[task_id]
            self._running_futures.pop(task_id, None)

        if tasks_to_remove:
            log.log_debug(f"Cleaned up {len(tasks_to_remove)} old tasks")

    def clear_completed(self):
        """Clear all completed, failed, and cancelled tasks."""
        with self._lock:
            tasks_to_remove = [
                tid for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                self._running_futures.pop(task_id, None)

            log.log_info(f"Cleared {len(tasks_to_remove)} completed tasks")

    def __len__(self) -> int:
        """Return the number of tasks."""
        with self._lock:
            return len(self._tasks)


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def reset_task_manager():
    """Reset the global task manager (for testing)."""
    global _task_manager
    _task_manager = None
