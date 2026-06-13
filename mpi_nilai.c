/*
 * ============================================================
 *  mpi_nilai.c
 *  Komputasi Paralel: Analisis Nilai Mahasiswa dengan OpenMPI
 *
 *  Arsitektur:
 *    - Rank 0 : Master Node  -> baca CSV, scatter, gather, output
 *    - Rank 1-4 : Worker Node -> hitung max/min/sum lokal
 *
 *  Kompilasi : mpicc -o mpi_nilai mpi_nilai.c
 *  Eksekusi  : mpirun --hostfile hosts.txt -np 5 ./mpi_nilai
 * ============================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>   /* DBL_MAX, DBL_MIN */

/* ── Konstanta ─────────────────────────────────────────── */
#define MAX_ROWS     200          /* kapasitas maksimum baris CSV      */
#define CSV_FILENAME "data_nilai.csv"

/* ── Fungsi: baca file CSV, kembalikan jumlah baris data ── */
int baca_csv(const char *filename, double *nilai_arr) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[Master] ERROR: File '%s' tidak ditemukan!\n", filename);
        return -1;
    }

    char line[128];
    int count = 0;

    /* Lewati baris header */
    fgets(line, sizeof(line), fp);

    /* Baca tiap baris: NIM,Nilai */
    while (fgets(line, sizeof(line), fp) && count < MAX_ROWS) {
        char *token = strtok(line, ",");  /* ambil kolom NIM (dibuang) */
        token = strtok(NULL, ",\n");      /* ambil kolom Nilai          */
        if (token) {
            nilai_arr[count++] = atof(token);
        }
    }
    fclose(fp);
    return count;
}

/* ══════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {

    int rank, size;

    /* [1] Inisialisasi MPI runtime */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* nomor proses ini   */
    MPI_Comm_size(MPI_COMM_WORLD, &size);   /* total proses aktif */

    /* ── Variabel bersama ─────────────────────────────── */
    double *semua_nilai  = NULL;   /* array penuh di Master          */
    double *lokal_nilai  = NULL;   /* potongan yang diterima Worker  */
    int     total_data   = 0;      /* jumlah baris valid di CSV      */
    int     lokal_count  = 0;      /* banyak data per Worker         */
    int    *sendcounts   = NULL;   /* berapa data dikirim ke tiap rank */
    int    *displs       = NULL;   /* offset tiap rank di array penuh */

    /* ── [2] MASTER: Baca CSV & hitung distribusi ─────── */
    if (rank == 0) {
        semua_nilai = (double *)malloc(MAX_ROWS * sizeof(double));
        if (!semua_nilai) { MPI_Abort(MPI_COMM_WORLD, 1); }

        total_data = baca_csv(CSV_FILENAME, semua_nilai);
        if (total_data <= 0) { MPI_Abort(MPI_COMM_WORLD, 1); }

        printf("[Master] Berhasil membaca %d baris data dari '%s'\n",
               total_data, CSV_FILENAME);
        printf("[Master] Membagi beban ke %d proses...\n", size);

        /* Hitung sendcounts & displs untuk MPI_Scatterv */
        sendcounts = (int *)malloc(size * sizeof(int));
        displs     = (int *)malloc(size * sizeof(int));

        int base  = total_data / size;   /* pembagian dasar          */
        int sisa  = total_data % size;   /* sisa baris (jika tidak habis) */

        for (int i = 0; i < size; i++) {
            /*
             * Rank 0..(sisa-1) mendapat 1 baris ekstra
             * sehingga semua data terdistribusi tanpa ada yang terlewat.
             */
            sendcounts[i] = base + (i < sisa ? 1 : 0);
            displs[i]     = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
        }
    }

    /* ── [3] Broadcast: beritahu semua proses jumlah lokal data ─ */
    /*
     * Setiap Worker perlu tahu berapa baris yang akan ia terima
     * sebelum MPI_Scatterv dipanggil.
     * Caranya: Scatter array sendcounts (satu int per rank).
     */
    MPI_Scatter(sendcounts, 1, MPI_INT,
                &lokal_count, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    /* ── [4] Alokasi buffer lokal di setiap proses ─────── */
    lokal_nilai = (double *)malloc(lokal_count * sizeof(double));
    if (!lokal_nilai) { MPI_Abort(MPI_COMM_WORLD, 1); }

    /* ── [5] SCATTER: Master kirim potongan data ke semua Worker ─ */
    MPI_Scatterv(
        semua_nilai,   /* buffer pengirim (hanya valid di rank 0) */
        sendcounts,    /* jumlah elemen untuk tiap rank           */
        displs,        /* offset awal tiap rank                   */
        MPI_DOUBLE,
        lokal_nilai,   /* buffer penerima di masing-masing rank   */
        lokal_count,
        MPI_DOUBLE,
        0,             /* root = Master                           */
        MPI_COMM_WORLD
    );

    /* ── [6] KOMPUTASI LOKAL: setiap proses hitung max/min/sum ── */
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

    /* ── [7] REDUCE: gabungkan hasil lokal ke Master ───── */
    double global_max, global_min, global_sum;

    MPI_Reduce(&lokal_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lokal_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lokal_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ── [8] OUTPUT AKHIR di Master ────────────────────── */
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

    /* [9] Finalisasi MPI */
    MPI_Finalize();
    return 0;
}
