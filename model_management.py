def pull_model(model_name, mo):
    # Taken from: https://github.com/ollama/ollama-python/blob/main/examples/pull.py
    from ollama import pull

    bars, completed_amounts = {}, {}

    for progress in pull(model_name, stream=True):
        digest = progress.get("digest", "")

        if not digest:
            mo.output.append(progress.get("status"))
            continue

        if digest not in bars and (total := progress.get("total")):
            # HACK: Call __enter__ to create the progress bar, to replicate how `for _ in mo.status.progress_bar(...)` works.
            bars[digest] = mo.status.progress_bar(total=total).__enter__()
            completed_amounts[digest] = 0

        if completed := progress.get("completed"):
            # Calculate the increment since last update
            last_completed = completed_amounts.get(digest, 0)
            increment = completed - last_completed
            if increment > 0:
                bars[digest].update(increment=increment)
                completed_amounts[digest] = completed

    # Clean up progress bars
    for bar in bars.values():
        try:
            bar.__exit__(None, None, None)
        except:
            pass
