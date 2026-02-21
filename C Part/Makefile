CC = mpicc
CFLAGS = -O2 -Wall -Wextra
LDFLAGS = -llapack -lblas -lm

tsqr: tsqr.c
	$(CC) $(CFLAGS) -o tsqr tsqr.c $(LDFLAGS)

clean:
	rm -f tsqr scaling_m.txt scaling_b.txt
