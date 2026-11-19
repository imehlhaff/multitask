"""
CPU version of compare_task_structures.py. The runs, data, task structures, steps and scoring are that script's (it is
imported, not copied, so changes there apply here too). Only two defaults differ:

  encoder        google/bert_uncased_L-4_H-256_A-4 (BERT-mini: 4 layers, hidden size 256, 11M parameters; Turc et al.
                 2019) instead of roberta-base (125M), which is too slow to fine-tune on CPU. For a stronger model at
                 about 4x the cost, pass --encoder google/bert_uncased_L-4_H-512_A-8 (BERT-small, 29M).
  learning_rate  1e-4. Small BERTs fine-tune best at higher rates than base-size models: Turc et al. pick from 3e-5 to
                 3e-4.

Every argument still overrides these. Results go to results/<encoder>_seed<seed>/ as before, so they sit apart from
the roberta-base runs, and `aggregate` here finds them because it uses the same default encoder.

Examples:
  python compare_task_structures_cpu.py tasks
  python compare_task_structures_cpu.py train --task 0
  python compare_task_structures_cpu.py aggregate
"""
import compare_task_structures as base

CPU_DEFAULTS = {
    "encoder": "google/bert_uncased_L-4_H-256_A-4",
    "learning_rate": 1e-4,
}

if __name__ == "__main__":
    # main() builds its parser from base.DEFAULTS and base.__doc__, so update both before calling it
    base.DEFAULTS.update(CPU_DEFAULTS)
    base.__doc__ = __doc__ + "\n" + "-" * 120 + "\n" + base.__doc__
    base.main()
