from collections import deque
from models import Task


class DAGengine:
    def __init__(self, tasks: list[Task]):
        self.tasks = {task.id: task for task in tasks}
        self.graph = {}
        self.in_degree = {}
        self._topo_order = []

        self._build_graph()
        self._compute_in_degrees()
        self._topo_order = self._run_kahn()

    def _build_graph(self):
        self.graph = {task_id: [] for task_id in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in self.graph:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep}'"
                    )
                self.graph[dep].append(task.id)

    def _compute_in_degrees(self):
        self.in_degree = {
            task_id: len(task.dependencies) for task_id, task in self.tasks.items()
        }

    def _run_kahn(self) -> list[str]:
        in_degrees = self.in_degree.copy()
        queue = deque(task_id for task_id, d in in_degrees.items() if d == 0)
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for successor in self.graph[node]:
                in_degrees[successor] -= 1
                if in_degrees[successor] == 0:
                    queue.append(successor)

        if len(order) != len(self.tasks):
            raise ValueError("Cycle detected in task dependencies")
        return order

    def topo_sort(self) -> list[str]:
        return self._topo_order

    def crit_path_length(self) -> float:
        dist: dict[str, float] = {
            task_id: self.tasks[task_id].duration
            for task_id in self.tasks
            if self.in_degree[task_id] == 0
        }

        for task_id in self._topo_order:
            current = dist.get(task_id, 0.0)
            for successor in self.graph[task_id]:
                candidate = current + self.tasks[successor].duration
                if candidate > dist.get(successor, 0.0):
                    dist[successor] = candidate

        return max(dist.values()) if dist else 0.0
