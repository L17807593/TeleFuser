from __future__ import annotations

import asyncio

from telefuser.service.livekit.session_registry import SessionRecord
from telefuser.service.livekit.worker_pool import InProcessLiveKitWorkerPool


class _CooperativeWorker:
    def __init__(self, *, complete_on_stop: bool) -> None:
        self.complete_on_stop = complete_on_stop
        self.running = asyncio.Event()
        self.stop_requested = asyncio.Event()
        self.completed = False
        self.cancelled = False

    async def start(self, *, skip_validation: bool = False) -> None:
        return None

    async def run_session(self, record: SessionRecord) -> None:
        del record
        self.running.set()
        try:
            await self.stop_requested.wait()
            if not self.complete_on_stop:
                await asyncio.Event().wait()
            self.completed = True
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def stop_session(self, session_id: str) -> None:
        del session_id
        self.stop_requested.set()

    async def stop(self) -> None:
        return None


def _record() -> SessionRecord:
    return SessionRecord(
        session_id="session-1",
        room_name="room-1",
        controller_identity="controller-1",
        status="assigned",
        worker_id="worker-0",
        config={},
        created_at=0,
        updated_at=0,
    )


def test_worker_pool_allows_cooperative_session_cleanup() -> None:
    async def _run() -> None:
        worker = _CooperativeWorker(complete_on_stop=True)
        pool = InProcessLiveKitWorkerPool({"worker-0": worker})
        await pool.start()
        pool.start_session(_record())
        await worker.running.wait()

        await pool.stop_session("session-1")

        assert worker.completed is True
        assert worker.cancelled is False

    asyncio.run(_run())


def test_worker_pool_cancels_session_after_cleanup_timeout(monkeypatch) -> None:
    async def _run() -> None:
        worker = _CooperativeWorker(complete_on_stop=False)
        pool = InProcessLiveKitWorkerPool({"worker-0": worker})
        await pool.start()
        pool.start_session(_record())
        await worker.running.wait()
        monkeypatch.setattr("telefuser.service.livekit.worker_pool._SESSION_STOP_GRACE_SECONDS", 0.01)

        await pool.stop_session("session-1")

        assert worker.completed is False
        assert worker.cancelled is True

    asyncio.run(_run())
