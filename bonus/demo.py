from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bonus.agent import HybridMemoryAgent  # noqa: E402


def seed_memories(agent: HybridMemoryAgent) -> None:
    memories = [
        "Tôi đã đọc một note về Kubernetes autoscaling: HPA tăng số pod theo CPU, còn cluster autoscaler tăng số node khi cluster thiếu capacity.",
        "Tài liệu cloud cost nói rằng AWS Lambda hợp với workload ngắn, nhưng service chạy liên tục có thể rẻ hơn trên Kubernetes nếu traffic ổn định.",
        "Một bài về cloud security nhấn mạnh zero-trust, OAuth, encryption at rest, và audit log cho hệ thống nhiều tenant.",
        "Tôi ghi chú rằng RAG cần embedding tốt, chunk vừa đủ nhỏ, và đánh giá hallucination bằng golden set thay vì chỉ nhìn demo đẹp.",
        "Tài liệu PostgreSQL giải thích replication, transaction ACID, index B-tree, và khi nào nên partition theo user_id.",
        "Tôi thích đọc tài liệu kỹ thuật bằng tiếng Việt trước, sau đó xem thuật ngữ tiếng Anh để đối chiếu khi cần implement.",
    ]
    for memory in memories:
        agent.remember(memory)


def main() -> None:
    agent = HybridMemoryAgent()
    seed_memories(agent)

    queries = [
        "Tôi đã đọc gì về Kubernetes?",
        "Recommend đọc gì tiếp",
        "Tôi đang quan tâm gì gần đây?",
        "Tài liệu về tự động mở rộng hạ tầng?",
        "Cho tôi summary cloud security",
    ]

    for i, query in enumerate(queries, start=1):
        print("=" * 88)
        print(f"Query {i}: {query}")
        print(agent.recall(query))


if __name__ == "__main__":
    main()
