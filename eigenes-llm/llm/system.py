"""Hilfen fuer lange Laeufe ueber Nacht."""

from __future__ import annotations

import contextlib
import sys

# Windows: SetThreadExecutionState-Flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def wach_bleiben():
    """Verhindert, dass Windows in den Energiesparmodus geht, solange der Block laeuft.

    Der Bildschirm darf trotzdem ausgehen. Nach dem Block (oder wenn das
    Programm endet) gelten wieder die normalen Energieeinstellungen.
    """
    aktiv = False
    if sys.platform == "win32":
        try:
            import ctypes

            aktiv = bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
            if aktiv:
                print("  Energiesparmodus ist waehrend des Laufs blockiert (Bildschirm darf ausgehen).")
        except Exception:
            aktiv = False
    try:
        yield aktiv
    finally:
        if aktiv:
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass
