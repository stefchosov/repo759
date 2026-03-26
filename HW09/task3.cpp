// HW09 Task 3: MPI point-to-point latency and bandwidth measurement.
// Usage: srun -n 2 ./task3 n
//   n -- number of floats in each message buffer
//
// Output (1 line):
//   ms  -- t0 + t1 in milliseconds (combined time for both processes)
//
// Compile: mpicxx task3.cpp -Wall -O3 -o task3
// Requires: module load mpi/mpich/4.0.2

#include <cstdlib>
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            std::cerr << "Error: must be run with exactly 2 processes." << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (argc != 2) {
        if (rank == 0)
            std::cerr << "Usage: srun -n 2 " << argv[0] << " n" << std::endl;
        MPI_Finalize();
        return 1;
    }

    const int n = std::atoi(argv[1]);

    float *buf_send = new float[n];
    float *buf_recv = new float[n];

    // Fill send buffer with arbitrary values.
    for (int i = 0; i < n; i++) buf_send[i] = static_cast<float>(i);

    MPI_Status status;

    if (rank == 0) {
        double t0_start = MPI_Wtime();
        MPI_Send(buf_send, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf_recv, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
        double t0 = MPI_Wtime() - t0_start;

        // Receive t1 from rank 1 and print total.
        double t1;
        MPI_Recv(&t1, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);

        std::cout << (t0 + t1) * 1000.0 << "\n";
    } else {
        double t1_start = MPI_Wtime();
        MPI_Recv(buf_recv, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(buf_send, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        double t1 = MPI_Wtime() - t1_start;

        // Send t1 to rank 0.
        MPI_Send(&t1, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    delete[] buf_send;
    delete[] buf_recv;

    MPI_Finalize();
    return 0;
}
