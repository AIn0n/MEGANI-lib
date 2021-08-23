#include "mx_ut.h"
#include "nn_ut.h"
#include <stdio.h>

int main (void)
{
	int failed = 0;
	failed += mx_ut();
	failed += nn_ut();

	puts("\nSUMMARY\n");
	if(failed)
		printf("%i ERRORS!\n", failed);
	else
		puts("NO ERRORS OCCURED!");

	return 0;
}