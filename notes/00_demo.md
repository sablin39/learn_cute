# Make sure everything is ready:

Assume you've followed my guidance in `README.md`, and is ready for a marvelous trip to `CuTe`. You can first try compiling the `demo.cu` to examine your environment. 

## Code Explained

This code is an example of performing GEMM with a template. 




### Notes on learning modern C++
- `decltype` and `auto`: 

    decltype gives the declared type of the expression that is passed to it. auto does the same thing as template type deduction. So, for example, if you have a function that returns a reference, auto will still be a value (you need auto& to get a reference), but decltype will be exactly the type of the return value.
    ```c++
        int global{};
        int& foo(){
            return global;
        }

        int main(){
            decltype(foo()) a = foo(); //a is an `int&`
            // decltype(auto)  a = foo();   alternatively, since C++14
            auto b = foo(); //b is an `int`
        }
    ```

- `static_assert` function: This performs an assert by triggering a **compile-time** error. 