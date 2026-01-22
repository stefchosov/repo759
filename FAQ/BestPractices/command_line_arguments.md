# Use Command Line Arguments Instead of Interactive Input

If a program you submit to a compute node requires inputs, you should pass
them with command line arguments, not keyboarding in interactively. When you work on a cluster like Euler, you can't be expected to type in arguments interactively.

In the past, some students ran their code on the head node directly, not because
they do not know the rules of using a cluster, but their Slurm jobs stall on a compute node, 
so they feel forced to try it interactively on the head node. This is possibly 
the result of writing a program that expects `std::cin` inputs.

---

##  Do Not Use C++ Interactive Input

It is NOT common to use C++ interactive input, such as
```c++
int N;
std::cin >> N;
```
in scientific computations. When your program is running on another node, 
you cannot touch it via `std::cin`. So it will  stall, waiting for
the input, until hits the job wall time limit and quits. Do not include similar 
lines in your code.

Instead, you should pass the inputs with command line arguments. 
Command line arguments are given after the name of the program when it is called.
We have tried to provide the call pattern for every HW problem that needs it, please 
follow it. An example is
```
./task N
```

---

## Command Line Arguments in C++

You can read about how to parse the arguments into your code on [this page](https://www.learncpp.com/cpp-tutorial/command-line-arguments/).

It can be useful to read the entire article but here is a brief summary:
1. Define your `main()` function with
```c++
int main(int argc, char *argv[]) {
    /* The content */
}
```
2. After the program is called, `argc` will be the total number of arguments 
(typically 1 or more, since the program name counts as the first argument), `argv`
will be the array of character pointers listing all the arguments.
3. Parse `argv` as needed. For example, if you would like to use `argv[1]` as an integer
```c++
int N = std::atoi(argv[1]);
```
or as  a floating point number
```c++
double N = std::stod(argv[1]);
```
or as a string  
```c++
std::string N(argv[1]);
```




