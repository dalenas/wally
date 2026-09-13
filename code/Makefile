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
	$(CXX) $(CXXFLAGS) -c src/interface/containers.cpp -o obj/interface/containers.o

obj/abstract/traits.o: src/abstract/traits.cpp lib/abstract/traits.h
	$(CXX) $(CXXFLAGS) -c src/abstract/traits.cpp -o obj/abstract/traits.o

obj/abstract/simd.o: src/abstract/simd.cpp lib/abstract/simd.h
	$(CXX) $(CXXFLAGS) -c src/abstract/simd.cpp -o obj/abstract/simd.o

obj/preprocessing/dataset.o: src/preprocessing/dataset.cpp lib/preprocessing/dataset.h
	$(CXX) $(CXXFLAGS) -c src/preprocessing/dataset.cpp -o obj/preprocessing/dataset.o

obj/preprocessing/stats.o: src/preprocessing/stats.cpp lib/preprocessing/stats.h
	$(CXX) $(CXXFLAGS) -c src/preprocessing/stats.cpp -o obj/preprocessing/stats.o

obj/models/ml.o: src/models/ml.cpp lib/models/ml.h
	$(CXX) $(CXXFLAGS) -c src/models/ml.cpp -o obj/models/ml.o

obj/models/linear_regression.o: src/models/linear_regression.cpp lib/models/linear_regression.h
	$(CXX) $(CXXFLAGS) -c src/models/linear_regression.cpp -o obj/models/linear_regression.o

obj/models/polynomial_regression.o: src/models/polynomial_regression.cpp lib/models/polynomial_regression.h
	$(CXX) $(CXXFLAGS) -c src/models/polynomial_regression.cpp -o obj/models/polynomial_regression.o

obj/models/binary_logistic_regression.o: src/models/binary_logistic_regression.cpp lib/models/binary_logistic_regression.h
	$(CXX) $(CXXFLAGS) -c src/models/binary_logistic_regression.cpp -o obj/models/binary_logistic_regression.o

obj/models/multi_logistic_regression.o: src/models/multi_logistic_regression.cpp lib/models/multi_logistic_regression.h
	$(CXX) $(CXXFLAGS) -c src/models/multi_logistic_regression.cpp -o obj/models/multi_logistic_regression.o

obj/models/model_library.o: src/models/model_library.cpp lib/models/model_library.h
	$(CXX) $(CXXFLAGS) -c src/models/model_library.cpp -o obj/models/model_library.o

obj/wally.o: src/wally.cpp lib/wally.h
	$(CXX) $(CXXFLAGS) -c src/wally.cpp -o obj/wally.o

clean:
	rm -f obj/interface/*.o
	rm -f obj/abstract/*.o
	rm -f obj/preprocessing/*.o
	rm -f obj/models/*.o
	rm -f obj/*.o