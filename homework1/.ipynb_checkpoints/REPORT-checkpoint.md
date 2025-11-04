
# Домашнее задание 1 по курсу LLM

## Эксперименты с гиперпараметрами
- **Batch size**: `per_device_train_batch_size ∈ {2, 4, 8}` и `gradient_accumulation_steps ∈ {2, 4, 8}`.
- **Learning rate**: `{3e-4, 4e-4, 5e-4}`.
- **Scheduler**: cosine/linear.

## Final evaluation results:
{'eval_loss': 4.55035924911499, 'eval_runtime': 44.0472, 'eval_samples_per_second': 113.515, 'eval_steps_per_second': 14.189, 'epoch': 0.0221188579345836}