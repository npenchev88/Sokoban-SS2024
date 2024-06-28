import modal

app = modal.App("example-get-started")


@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    rez = square.remote(42)
    output_file = "result.txt"
    with open(output_file, 'w') as f:
        f.write(str(rez))
    print("the square is", rez)
