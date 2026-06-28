"""
P0 driver: run TASK #3 (multi-seed) then TASK #4 (theory profiling) in ONE
process so Klein-9B is loaded exactly once (eval_all is imported once and cached;
both run_seed and profile_theory reuse the same loaded module).

Usage:
  /home/ubuntu/miniconda3/envs/sensenova/bin/python scripts/run_p0.py
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_seed          # triggers the single eval_all / Klein-9B load
import profile_theory    # reuses the already-loaded eval_all module

if __name__ == "__main__":
    t0 = time.time()
    print("\n##### TASK #3: multi-seed significance #####", flush=True)
    run_seed.main((44, 45))
    print("\n##### TASK #4: theory assumption profiling #####", flush=True)
    profile_theory.main()
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min")
