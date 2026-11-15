from __future__ import annotations

import json

from .r5a3b_p2a_transport_pilot import ARTIFACT_ROOT, main


if __name__ == "__main__":
    report = main()
    print(json.dumps({"artifact_root": str(ARTIFACT_ROOT), "final_report": report}, indent=2, sort_keys=True))
