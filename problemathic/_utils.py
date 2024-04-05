import time

def retry(max_retries, sleep_time):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying...")
                    time.sleep(sleep_time)
            else:
                return {}, Exception(f"Max retries of {max_retries} exceeded for function {func.__name__}")
        return wrapper
    return decorator


@retry(max_retries=3, sleep_time=10)
def chain_runner(self, chain, **kwargs):
    """
    Function to invoke chains
    """
    output = chain.invoke(**kwargs)
    return output
