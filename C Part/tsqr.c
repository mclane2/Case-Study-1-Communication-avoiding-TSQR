/*
 *
 * Case 1: Question 2+3
 * Communication-Avoiding Tall Skinny QR (TSQR) Algorithm
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// LAPACK and BLAS prototypes
void dgeqrf_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
void dorgqr_(int *m, int *n, int *k, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);

// Function prototypes
// For QR factorisation using LAPACK routines:
void qr_factorize(double *A, int rows, int cols, double *Q, double *R);
// Vertically stacks two matrices:
void stack_matrices(double *top, int top_rows, double *bot, int bot_rows, int cols, double *result);
// Splits matrix into blocks of rows for sending to different processors:
void extract_rows(double *src, int src_rows, int row_start, int num_rows, int cols, double *dst);
// Implements the full TSQR algorithm:
void run_tsqr(int m, int b, int rank, double *time_out, double *recon_err, double *orth_err);


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Testing accuracy on 200 by 5 random matrix
    double time_out, recon_err, orth_err;
    run_tsqr(200, 5, rank, &time_out, &recon_err, &orth_err);
    if (rank == 0) {
        printf("Accuracy test (m=200, b=5)\n");
        printf("||W - QR||  = %e\n", recon_err);
        printf("||Q'Q - I|| = %e\n", orth_err);
        printf("Time: %f s\n\n", time_out);
    }

    // Testing scaling with respect to m
    if (rank == 0) printf("Scaling: varying m (b=5)\n");
    FILE *f_m = NULL;
    if (rank == 0) f_m = fopen("scaling_m.txt", "w");
    if (rank == 0) fprintf(f_m, "m,b,time\n");

    int b_fixed = 5;
    int m_values[] = {100, 500, 1000, 5000, 10000, 50000, 100000};
    int n_m = sizeof(m_values) / sizeof(m_values[0]);

    // Running the algorithm for each value of m
    for (int i = 0; i < n_m; i++) {
        run_tsqr(m_values[i], b_fixed, rank, &time_out, &recon_err, &orth_err);
        if (rank == 0) {
            fprintf(f_m, "%d,%d,%e\n", m_values[i], b_fixed, time_out);
            printf("m=%6d  b=%3d  time=%e  error=%e\n", m_values[i], b_fixed, time_out, recon_err);
        }
    }
    if (rank == 0) fclose(f_m);

    // Scaling with respect to b
    if (rank == 0) printf("\n Scaling: varying b (m=10000)\n");
    FILE *f_b = NULL;
    if (rank == 0) f_b = fopen("scaling_b.txt", "w");
    if (rank == 0) fprintf(f_b, "m,b,time\n");

    int m_fixed = 10000;
    int b_values[] = {2, 5, 10, 20, 50, 100, 200};
    int n_b = sizeof(b_values) / sizeof(b_values[0]);

    // Running the algorithm for each value of b
    for (int i = 0; i < n_b; i++) {
        run_tsqr(m_fixed, b_values[i], rank, &time_out, &recon_err, &orth_err);
        if (rank == 0) {
            fprintf(f_b, "%d,%d,%e\n", m_fixed, b_values[i], time_out);
            printf("m=%6d  b=%3d  time=%e  error=%e\n", m_fixed, b_values[i], time_out, recon_err);
        }
    }
    if (rank == 0) fclose(f_b);

    MPI_Finalize();
    return 0;
}

/*
 * Run the full TSQR algorithm for an (m x b) random matrix.
 * Returns wall-clock time and error
 */
void run_tsqr(int m, int b, int rank, double *time_out, double *recon_err, double *orth_err) {
    int block_size = m / 4;

    // Generating the same random matrix on every processor
    srand(42);
    double *W_full = malloc(m * b * sizeof(double));
    for (int j = 0; j < b; j++)
        for (int i = 0; i < m; i++)
            W_full[i + j * m] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

    // Each rank extracts its own row block
    double *W_local = malloc(block_size * b * sizeof(double));
    extract_rows(W_full, m, rank * block_size, block_size, b, W_local);

    // Blocking to keep track of scaling times
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Stage 1: Implementing local QR factorisation on each block
    double *Q_local = malloc(block_size * b * sizeof(double));
    double *R_local = malloc(b * b * sizeof(double));
    qr_factorize(W_local, block_size, b, Q_local, R_local);

    // Stage 2: Sending rank 1's R to rank 0 and sending rank 3's R to rank 2
    double *R_recv    = malloc(b * b * sizeof(double));
    double *R_stacked = malloc(2 * b * b * sizeof(double));
    double *Q_stage2  = NULL;
    double *R_stage2  = malloc(b * b * sizeof(double));

    if (rank == 1) {
        MPI_Send(R_local, b * b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        MPI_Recv(R_recv, b * b, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stack_matrices(R_local, b, R_recv, b, b, R_stacked);
        Q_stage2 = malloc(2 * b * b * sizeof(double));
        qr_factorize(R_stacked, 2 * b, b, Q_stage2, R_stage2);
    }
    if (rank == 3) {
        MPI_Send(R_local, b * b, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
    }
    if (rank == 2) {
        MPI_Recv(R_recv, b * b, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stack_matrices(R_local, b, R_recv, b, b, R_stacked);
        Q_stage2 = malloc(2 * b * b * sizeof(double));
        qr_factorize(R_stacked, 2 * b, b, Q_stage2, R_stage2);
    }

    // Stage 3: Sending rank 2's contents to rank 0
    double *Q_stage3 = NULL;
    double *R_final  = malloc(b * b * sizeof(double));

    if (rank == 2) {
        MPI_Send(R_stage2, b * b, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(Q_stage2, 2 * b * b, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        double *R_from_2 = malloc(b * b * sizeof(double));
        MPI_Recv(R_from_2, b * b, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        stack_matrices(R_stage2, b, R_from_2, b, b, R_stacked);
        Q_stage3 = malloc(2 * b * b * sizeof(double));
        qr_factorize(R_stacked, 2 * b, b, Q_stage3, R_final);
        free(R_from_2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    *time_out = MPI_Wtime() - t_start;

    // Reconstructing Q on rank 0: Getting ranks 1, 2, 3 send their Q_local to rank 0
    if (rank >= 1) {
        MPI_Send(Q_local, block_size * b, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }

    *recon_err = 0.0;
    *orth_err  = 0.0;

    if (rank == 0) {
        double *Q_11 = malloc(2 * b * b * sizeof(double));
        MPI_Recv(Q_11, 2 * b * b, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *Q_all[4];
        Q_all[0] = Q_local;
        for (int r = 1; r <= 3; r++) {
            Q_all[r] = malloc(block_size * b * sizeof(double));
            MPI_Recv(Q_all[r], block_size * b, MPI_DOUBLE, r, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Splitting Q_02 (2b x b) into top and bottom halves, each (b x b)
        double *Q_02_top = malloc(b * b * sizeof(double));
        double *Q_02_bot = malloc(b * b * sizeof(double));
        extract_rows(Q_stage3, 2 * b, 0, b, b, Q_02_top);
        extract_rows(Q_stage3, 2 * b, b, b, b, Q_02_bot);

        double *mid_top = malloc(2 * b * b * sizeof(double));
        double *mid_bot = malloc(2 * b * b * sizeof(double));
        double alpha = 1.0, beta = 0.0;
        int twob = 2 * b;
        dgemm_("N", "N", &twob, &b, &b, &alpha, Q_stage2, &twob, Q_02_top, &b, &beta, mid_top, &twob);
        dgemm_("N", "N", &twob, &b, &b, &alpha, Q_11, &twob, Q_02_bot, &b, &beta, mid_bot, &twob);

        // Splitting into 4 pieces, each (b x b)
        double *mid[4];
        for (int k = 0; k < 4; k++) mid[k] = malloc(b * b * sizeof(double));
        extract_rows(mid_top, twob, 0, b, b, mid[0]);
        extract_rows(mid_top, twob, b, b, b, mid[1]);
        extract_rows(mid_bot, twob, 0, b, b, mid[2]);
        extract_rows(mid_bot, twob, b, b, b, mid[3]);

        double *Q_full = malloc(m * b * sizeof(double));
        for (int k = 0; k < 4; k++) {
            double *Q_block = malloc(block_size * b * sizeof(double));
            dgemm_("N", "N", &block_size, &b, &b, &alpha, Q_all[k], &block_size, mid[k], &b, &beta, Q_block, &block_size);
            for (int j = 0; j < b; j++)
                for (int i = 0; i < block_size; i++)
                    Q_full[(k * block_size + i) + j * m] = Q_block[i + j * block_size];
            free(Q_block);
        }

        // Verifying ||W - QR||
        double *QR = malloc(m * b * sizeof(double));
        dgemm_("N", "N", &m, &b, &b, &alpha,
               Q_full, &m, R_final, &b, &beta, QR, &m);
        double norm = 0.0;
        for (int i = 0; i < m * b; i++) {
            double diff = W_full[i] - QR[i];
            norm += diff * diff;
        }
        *recon_err = sqrt(norm);

        // Verifying ||Q'Q - I|| 
        double *QtQ = malloc(b * b * sizeof(double));
        dgemm_("T", "N", &b, &b, &m, &alpha,
               Q_full, &m, Q_full, &m, &beta, QtQ, &b);
        double oerr = 0.0;
        for (int j = 0; j < b; j++)
            for (int i = 0; i < b; i++) {
                double val = QtQ[i + j * b] - (i == j ? 1.0 : 0.0);
                oerr += val * val;
            }
        *orth_err = sqrt(oerr);

        // Cleaning up rank 0
        free(QR); free(QtQ); free(Q_full);
        free(Q_02_top); free(Q_02_bot);
        free(mid_top); free(mid_bot);
        for (int k = 0; k < 4; k++) free(mid[k]);
        for (int r = 1; r <= 3; r++) free(Q_all[r]);
        free(Q_11);
    }

    // Cleaning up all ranks
    free(W_full); free(W_local); free(Q_local); free(R_local); free(R_recv); free(R_stacked); free(R_stage2); free(R_final);
    if (Q_stage2) free(Q_stage2);
    if (Q_stage3) free(Q_stage3);
}

/*
 * QR-factorise a column-major matrix A.
 * Q is (rows x cols), R is (cols x cols).
 */
void qr_factorize(double *A, int rows, int cols, double *Q, double *R) {
    // Copying A into Q because dgeqrf overwrites its input
    memcpy(Q, A, rows * cols * sizeof(double));

    double *tau = malloc(cols * sizeof(double));
    int lwork, info;
    double work_query;

    // Setting up dgeqrf function
    lwork = -1;
    dgeqrf_(&rows, &cols, Q, &rows, tau, &work_query, &lwork, &info);
    lwork = (int)work_query;
    double *work = malloc(lwork * sizeof(double));

    // Compute QR (R stored in upper triangle of Q)
    dgeqrf_(&rows, &cols, Q, &rows, tau, work, &lwork, &info);

    // Extracting R from the upper triangular
    memset(R, 0, cols * cols * sizeof(double));
    for (int j = 0; j < cols; j++)
        for (int i = 0; i <= j; i++)
            R[i + j * cols] = Q[i + j * rows];

    // Forming explicit Q from the Householder reflectors
    free(work);
    lwork = -1;
    dorgqr_(&rows, &cols, &cols, Q, &rows, tau, &work_query, &lwork, &info);
    lwork = (int)work_query;
    work = malloc(lwork * sizeof(double));
    dorgqr_(&rows, &cols, &cols, Q, &rows, tau, work, &lwork, &info);

    free(tau);
    free(work);
}

/*
 * Stack two column-major matrices vertically:
 */
void stack_matrices(double *top, int top_rows, double *bot, int bot_rows, int cols, double *result) {
    int total_rows = top_rows + bot_rows;
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < top_rows; i++)
            result[i + j * total_rows] = top[i + j * top_rows];
        for (int i = 0; i < bot_rows; i++)
            result[(i + top_rows) + j * total_rows] = bot[i + j * bot_rows];
    }
}

/*
 * Extract rows [row_start, row_start + num_rows) from an (src_rows x cols)
 */
void extract_rows(double *src, int src_rows, int row_start, int num_rows, int cols, double *dst) {
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < num_rows; i++)
            dst[i + j * num_rows] = src[(row_start + i) + j * src_rows];
}







