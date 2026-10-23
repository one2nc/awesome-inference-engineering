# Contribution Guidelines

This list is curated, not exhaustive. Only add resources that you would recommend to an engineer working on inference.

## Adding a Resource

- Search the list first to avoid duplicates. Each resource appears only once, in its best-fit section.
- Use the format `- [Name](link) - Description.`
- Keep descriptions short and factual. Start with a capital letter and end with a period.
- For papers, add the venue or year at the end of the description, for example `(OSDI 2024)`.
- Add new entries at the bottom of the section, unless the entry is more introductory than what is already there.
- Link to the primary source: the upstream repository, official docs or the paper itself.
- Use one pull request per resource, and explain why it is useful for inference engineering.

## What Belongs Here

- Tools, papers, articles, talks and courses about running models in production: engines, kernels, caching, scheduling, distributed serving, hardware, observability and benchmarking.
- Material with real technical depth, such as code, documentation, papers or reproducible experiments.

## What Does Not Belong Here

- Training, fine-tuning or agent frameworks, unless the resource is mainly about inference.
- Marketing posts and speed claims without a reproducible method.
- Archived or unmaintained projects. Mark a project as being in maintenance mode when that applies.
- Release notes, individual pull requests and social media posts.

## Benchmarks

When a resource reports performance numbers, it should state the hardware, model, precision, engine version, input and output lengths, concurrency, and when it was measured. Vendor claims must not be presented as independent results.
