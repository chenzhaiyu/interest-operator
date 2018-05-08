#include "mymath.h"

int min(int x1, int x2, int x3, int x4)
{
	return (x1 < x2 ? x1 : x2) < (x3 < x4 ? x3 : x4) ? (x1 < x2 ? x1 : x2) : (x3 < x4 ? x3 : x4);
}