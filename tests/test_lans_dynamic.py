import os
import sys

sys.path.append(os.getcwd())

from oglans.utils.ds_cns import LANSScheduler


def test_lans_dynamics():
    print("🧪 Testing LANS Dynamics...")
    scheduler = LANSScheduler(d_max=4, d_min=1, loss_baseline=0.5, ema_decay=0.8)

    print("\n[Phase 1] Simulating High Loss (Bad Performance)...")
    for _ in range(5):
        comp = scheduler.update_competence(batch_loss=10.0)
        thresh = scheduler.current_threshold
        strategy = scheduler.get_strategy()
        print(f"Loss=10.0 -> Competence={comp:.4f}, Threshold={thresh:.2f}, Strategy={strategy}")

    print("\n[Phase 2] Simulating Low Loss (Good Performance)...")
    for _ in range(10):
        comp = scheduler.update_competence(batch_loss=0.01)
        thresh = scheduler.current_threshold
        strategy = scheduler.get_strategy()
        print(f"Loss=0.01 -> Competence={comp:.4f}, Threshold={thresh:.2f}, Strategy={strategy}")

    if comp > 0.5:
        print("\n✅ LANS Scheduler reacts correctly to loss.")
    else:
        print("\n❌ LANS Scheduler failed to increase competence.")


if __name__ == "__main__":
    test_lans_dynamics()
