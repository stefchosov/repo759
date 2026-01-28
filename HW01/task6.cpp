#include <cstdio>
#include <cstdlib>
#include <iostream>
int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }

    int N = std::atoi(argv[1]);
    for (int i = 0; i <= N; ++i) {
        std::printf("%d", i);
        if (i < N) std::printf(" ");
    }
    std::printf("\n");
    for (int i = N; i >= 0; --i) {
        std::cout << i;
        if (i > 0) std::cout << " ";
    }
    std::cout << "\n";

    return 0;
}
