import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(f"""This is an experiment to understand how multi-threading works in Marimo.""")
    return


@app.cell
def _():
    import threading
    import time
    import queue
    import random
    return queue, random, threading, time


@app.cell
def _(queue, random, threading, time):
    # Create a queue for thread-safe printing
    print_queue = queue.Queue()

    def worker(name, sleep_time):
        """Simulates a long-running process"""
        print_queue.put(f"Thread {name} starting...")
        time.sleep(sleep_time)  # Simulate work
        print_queue.put(f"Thread {name} finished after {sleep_time} seconds")

    # Create three threads with different sleep times
    threads = [
        threading.Thread(target=worker, args=(f"Worker-{i}", random.randint(1, 8)))
        for i in range(1, 4)
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Print all messages in order
    messages = []
    while not print_queue.empty():
        messages.append(print_queue.get())

    return (messages,)


@app.cell
def _(messages):
    messages
    return


if __name__ == "__main__":
    app.run()
