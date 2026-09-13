CXX = g++
CXXFLAGS = -Wall -Wextra -fsanitize=address -std=c++20 -O3 -mavx2 -mfma -Ilib/

SIMD: obj/interface/containers.o obj/abstract/traits.o obj/abstract/simd.o
PREPROCESS: $(SIMD) obj/preprocessing/dataset.o obj/preprocessing/stats.o

ML: obj/models/ml.o
REGRESSORS: $(ML) obj/models/linear_regression.o obj/models/polynomial_regression.o
CLASSIFIERS: $(ML) obj/models/binary_logistic_regression.o obj/models/multi_logistic_regression.o
MODELS: $(REGRESSORS) $(CLASSIFIERS) obj/models/model_library.o

all: $(PREPROCESS) $(MODELS) obj/wally.o

obj/interface/containers.o: src/interface/containers.cpp lib/interface/containers.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/abstract/traits.o: src/abstract/traits.cpp lib/abstract/traits.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/abstract/simd.o: src/abstract/simd.cpp lib/abstract/simd.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/preprocessing/dataset.o: src/preprocessing/dataset.cpp lib/preprocessing/dataset.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/preprocessing/stats.o: src/preprocessing/stats.cpp lib/preprocessing/stats.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/models/ml.o: src/models/ml.cpp lib/models/ml.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/models/linear_regression.o: src/models/linear_regression.cpp lib/models/linear_regression.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/models/polynomial_regression.o: src/models/polynomial_regression.cpp lib/models/polynomial_regression.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/models/binary_logistic_regression.o: src/models/binary_logistic_regression.cpp lib/models/binary_logistic_regression.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/models/multi_logistic_regression.o: src/models/multi_logistic_regression.cpp lib/models/multi_logistic_regression.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/models/model_library.o: src/models/model_library.cpp lib/models/model_library.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

obj/wally.o: src/wally.cpp lib/wally.h
	$(CXX) $(CXXFLAGS) -c $@ -o $<

clean:
	rm -f obj/interface/*.o
	rm -f obj/abstract/*.o
	rm -f obj/preprocessing/*.o
	rm -f obj/models/*.o
	rm -f obj/*.o