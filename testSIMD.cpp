#include <xmmintrin.h>
#include <stdio.h>
#include <chrono>
#include <process.h>

using namespace std::chrono;

__declspec(align(16)) float mat0[] = { 2.0f, -1.0f, 3.0f, 1.0f };
__declspec(align(16)) float mat1[] = { 1.0f, 3.0f, 1.0f, 0.0f };
__declspec(align(16)) float mat2[] = { -2.0f, 2.0f, -1.0f, 0.0f };
__declspec(align(16)) float mat3[] = { 3.0f, -1.0f, 0.0f, 1.0f };
__declspec(align(16)) float vec0[] = { -2.0f, 1.0f, -3.0f, 1.0f };
__declspec(align(16)) float vec1[] = { -2.0f, 1.0f, -3.0f, 1.0f };

int main(int argc, char* argv[]) {

	int index = 0;
	int loopCount = 10000000;

	auto startTime1 = high_resolution_clock::now();
	for (index = 0; index < loopCount; ++index) {
		vec1[0] = 0;
		vec1[1] = 0;
		vec1[2] = 0;
		vec1[3] = 0;
		vec1[0] += mat0[0] * vec0[0];
		vec1[1] += mat0[1] * vec0[1];
		vec1[2] += mat0[2] * vec0[2];
		vec1[3] += mat0[3] * vec0[3];
		vec1[0] += mat1[0] * vec0[0];
		vec1[1] += mat1[1] * vec0[1];
		vec1[2] += mat1[2] * vec0[2];
		vec1[3] += mat1[3] * vec0[3];
		vec1[0] += mat2[0] * vec0[0];
		vec1[1] += mat2[1] * vec0[1];
		vec1[2] += mat2[2] * vec0[2];
		vec1[3] += mat2[3] * vec0[3];
		vec1[0] += mat3[0] * vec0[0];
		vec1[1] += mat3[1] * vec0[1];
		vec1[2] += mat3[2] * vec0[2];
		vec1[3] += mat3[3] * vec0[3];
	}
	auto endTime1 = high_resolution_clock::now();
	int totalTimeSec1 = (endTime1 - startTime1).count() / 1e6;
	printf("Mat * Vec: %g %g %g %g\n", vec1[0], vec1[1], vec1[2], vec1[3]);
	printf("NOT SIMD Complete No:%d, Use Time(MilliSec):%d\n", loopCount, totalTimeSec1);

	auto startTime0 = high_resolution_clock::now();
	__m128 result0, result1, result2, result3;
	for (index = 0; index < loopCount; ++index) {
		__m128 vec = _mm_load_ps(vec0);
		__m128 xRow = _mm_load_ps(mat0);
		__m128 yRow = _mm_load_ps(mat1);
		__m128 zRow = _mm_load_ps(mat2);
		__m128 wRow = _mm_load_ps(mat3);
		result0 = _mm_mul_ps(vec, xRow);
		result1 = _mm_mul_ps(vec, yRow);
		result2 = _mm_mul_ps(vec, zRow);
		result3 = _mm_mul_ps(vec, wRow);
		_mm_store_ps(vec1, _mm_add_ps(result0, _mm_add_ps(result1, _mm_add_ps(result2, result3))));
	}
	auto endTime0 = high_resolution_clock::now();
	int totalTimeSec0 = (endTime0 - startTime0).count() / 1e6;
	printf("Mat * Vec: %g %g %g %g\n", vec1[0], vec1[1], vec1[2], vec1[3]);
	printf("SIMD Complete No:%d, Use Time(MilliSec):%d\n", loopCount, totalTimeSec0);

	system("pause");
	return 0;
}