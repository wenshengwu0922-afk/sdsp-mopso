# Linux/macOS build (Windows: run 00_build.ps1, which builds ssdsp.exe)
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -pthread

ssdsp: src/*.hpp src/main.cpp
	$(CXX) $(CXXFLAGS) src/main.cpp -o $@

test: ssdsp
	./ssdsp --exact --case stapler --out /tmp/ssdsp_exact_stapler | grep -q "HV = 0.831917" && echo "stapler exact front OK"
	./ssdsp --exact --case fan --out /tmp/ssdsp_exact_fan | grep -q "HV = 0.859667" && echo "fan exact front OK"
	./ssdsp --alg MOPSO --case fan --budget 20000 --seeds 5001-5003 --set "w=0,c1=3.5,c2=4.5" --threads 3 --out /tmp/ssdsp_mopso

clean:
	rm -f ssdsp ssdsp.exe

.PHONY: test clean
