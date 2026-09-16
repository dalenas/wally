CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O3 -mavx2 -mfma -Ilib -Isrc

TEST_LIBS = -lgtest
BENCHMARK_LIBS = -lbenchmark

SIMD: obj\abstract\traits.o obj\abstract\simd.o
PREPROCESS: $(SIMD) obj\preprocessing\dataset.o obj\preprocessing\stats.o

ML: obj\models\ml.o
REGRESSORS: $(ML) obj\models\linear_regression.o obj\models\polynomial_regression.o
CLASSIFIERS: $(ML) obj\models\binary_logistic_regression.o obj\models\multi_logistic_regression.o
MODELS: $(REGRESSORS) $(CLASSIFIERS) obj\models\model_library.o

temp: $(SIMD) test\obj\test_containers.o test\test_main.exe

obj\abstract\traits.o: lib\abstract\traits.h | obj\abstract
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\preprocessing\dataset.o: src\preprocessing\dataset.cpp lib\preprocessing\dataset.h | obj\preprocessing
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\preprocessing\stats.o: src\preprocessing\stats.cpp lib\preprocessing\stats.h | obj\preprocessing
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\models\ml.o: src\models\ml.cpp lib\models\ml.h | obj\models
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\models\linear_regression.o: src\models\linear_regression.cpp lib\models\linear_regression.h | obj\models
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\models\polynomial_regression.o: src\models\polynomial_regression.cpp lib\models\polynomial_regression.h | obj\models
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\models\binary_logistic_regression.o: src\models\binary_logistic_regression.cpp lib\models\binary_logistic_regression.h | obj\models
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\models\multi_logistic_regression.o: src\models\multi_logistic_regression.cpp lib\models\multi_logistic_regression.h | obj\models
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\models\model_library.o: src\models\model_library.cpp lib\models\model_library.h | obj\models
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj\wally.o: src\wally.cpp lib\wally.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

test\obj\test_containers.o: test\src\test_containers.cpp lib\interface\containers.h src\interface\containers.tpp | test\obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

test\obj\test_simd.o: test\src\test_simd.cpp lib\abstract\simd.h src\abbstract\simd.tpp | test\obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

test\obj\test_main.o: test\src\test_main.cpp | test\obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

test\test_main.exe: test\obj\test_main.o test\obj\test_containers.o
	$(CXX) $(CXXFLAGS) $^ $(TEST_LIBS) -o $@

obj\abstract:
	mkdir $@

obj\interface:
	mkdir $@

obj\models:
	mkdir $@

obj\preprocessing:
	mkdir $@

test\obj:
	mkdir $@

clean:
	-del obj\interface\*.o
	-del obj\abstract\*.o
	-del obj\preprocessing\*.o
	-del obj\models\*.o
	-del obj\*.o
	-del test\obj\*.o
	-del test\*.exe
	-rmdir obj\interface
	-rmdir obj\abstract
	-rmdir obj\preprocessing
	-rmdir obj\models
	-rmdir obj
	-rmdir test\obj
	-rmdir test