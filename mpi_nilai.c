#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>   

#define MAX_ROWS     200          /* kapasitas maksimum baris CSV      */
#define CSV_FILENAME "data_nilai.csv"

int baca_csv(const char *filename, double *nilai_arr) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[Master] ERROR: File '%s' tidak ditemukan!\n", filename);
        return -1;
    }

    char line[128];
    int count = 0;

    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp) && count < MAX_ROWS) {
        char *token = strtok(line, ",");  
        token = strtok(NULL, ",\n");     
        if (token) {
            nilai_arr[count++] = atof(token);
        }
    }
    fclose(fp);
    return count;
}

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    MPI_Comm_size(MPI_COMM_WORLD, &size);   

    double *semua_nilai  = NULL;  
    double *lokal_nilai  = NULL;  
    int     total_data   = 0;      
    int     lokal_count  = 0;     
    int    *sendcounts   = NULL;   
    int    *displs       = NULL;  

    if (rank == 0) {
        semua_nilai = (double *)malloc(MAX_ROWS * sizeof(double));
        if (!semua_nilai) { MPI_Abort(MPI_COMM_WORLD, 1); }

        total_data = baca_csv(CSV_FILENAME, semua_nilai);
        if (total_data <= 0) { MPI_Abort(MPI_COMM_WORLD, 1); }

        printf("[Master] Berhasil membaca %d baris data dari '%s'\n",
               total_data, CSV_FILENAME);
        printf("[Master] Membagi beban ke %d proses...\n", size);

        sendcounts = (int *)malloc(size * sizeof(int));
        displs     = (int *)malloc(size * sizeof(int));

        int base  = total_data / size;   
        int sisa  = total_data % size;   

        for (int i = 0; i < size; i++) {
            sendcounts[i] = base + (i < sisa ? 1 : 0);
            displs[i]     = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
        }
    }

    MPI_Scatter(sendcounts, 1, MPI_INT,
                &lokal_count, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    lokal_nilai = (double *)malloc(lokal_count * sizeof(double));
    if (!lokal_nilai) { MPI_Abort(MPI_COMM_WORLD, 1); }

    MPI_Scatterv(
        semua_nilai,   
        sendcounts,  
        displs,        
        MPI_DOUBLE,
        lokal_nilai,   
        lokal_count,
        MPI_DOUBLE,
        0,             
        MPI_COMM_WORLD
    );

    double lokal_max = -DBL_MAX;
    double lokal_min =  DBL_MAX;
    double lokal_sum = 0.0;

    for (int i = 0; i < lokal_count; i++) {
        if (lokal_nilai[i] > lokal_max) lokal_max = lokal_nilai[i];
        if (lokal_nilai[i] < lokal_min) lokal_min = lokal_nilai[i];
        lokal_sum += lokal_nilai[i];
    }

    printf("[Rank %d] lokal_count=%d | max=%.1f | min=%.1f | sum=%.2f\n",
           rank, lokal_count, lokal_max, lokal_min, lokal_sum);

    double global_max, global_min, global_sum;

    MPI_Reduce(&lokal_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lokal_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lokal_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rata_rata = global_sum / total_data;

        printf("\n===========================================\n");
        printf("  HASIL ANALISIS NILAI MAHASISWA (MPI)\n");
        printf("===========================================\n");
        printf("  Total Data Diproses : %d baris\n", total_data);
        printf("  Nilai Tertinggi     : %.1f\n", global_max);
        printf("  Nilai Terendah      : %.1f\n", global_min);
        printf("  Rata-rata Kelas     : %.2f\n", rata_rata);
        printf("===========================================\n");

        free(semua_nilai);
        free(sendcounts);
        free(displs);
    }

    free(lokal_nilai);

    MPI_Finalize();
    return 0;
}
