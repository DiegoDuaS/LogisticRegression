from memory_profiler import memory_usage
import time

def time_and_memory(model, X_train, y_train, X_test, title):
    training(model, X_train, y_train, title)
    predicting(model, X_test, title)

def training(model, X_train, y_train, title):
    start = time.time()
    mem_usage = memory_usage((model.fit, (X_train, y_train)), max_usage=True)
    end = time.time()
    print(f"Training {title}: ")
    print(f"- Tiempo: {end - start:.2f}s")
    print(f"- Memoria Peak: {mem_usage} MiB")
    
def predicting(model, X_test, title):
    start = time.time()
    mem_usage = memory_usage((model.predict, (X_test,)), max_usage=True)
    end = time.time()
    print(f"Predicting {title}: ")
    print(f"- Tiempo: {end - start:.2f}s")
    print(f"- Memoria Peak: {mem_usage} MiB")