#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <random>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//Wichtig für die Verwendung von Pybind11: Umstellen des Konfigurationstyps auf den
//dynamic loaded library (.dll) - Typ
//Für konkrete Einstellungen,
//s. https://docs.microsoft.com/de-de/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2019

//Zuweisung des namespace für Komfortabilität in der anschließenden Verwendung.
namespace py = pybind11;

// ----------------------------------------------------------------------------

/*
Bekannte Hilfsfunktion, diesmal mit einem 2D Vektor, anstatt einem Array
*/

void printGrid(std::vector<std::vector<int>> &grid) {
    int N = grid.size();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << " " << grid[i][j] << " ";
        }
        std::cout << "\n";
    }
}

/*
Bekannte Hilfsfunktion
*/

int trueValue(int x, int N) {
    return (x + N) % N;
}

/*
Aus der C++ bekannte bekannte Hilfsfunktion,
diesmal mit einem 2D Vektor, anstatt einem Array.
Übergabe wieder als Referenz.
*/

int neighbors(int y, int x, std::vector<std::vector<int>> &grid) {
    int count = 0;
    int N = grid.size();
    int range[] = { -1, 0, 1 };
    for (int v : range) {
        for (int w : range) {
            if (w || v) {
                count += grid[trueValue(y + w, N)][trueValue(x + v, N)];
            }
        }
    }
    return count;
}

/*
Diese Funktion führt ein Game of Life auf einem als Referenz-Parameter gegebenen (2D-)Vektor aus.
Ein Vektor ist ein dynamisches Array. Es gleicht damit ein wenig der Liste in Python.
Der Funktion wird außerdem der Paramter der durchzuführenden Iterationen übergeben.
Die Anzahl der durchzuführenden Iterationen des Game of Life werden hier aus Python in
C++ verlegt. (Die Funktion war allerdings schon vorher weit schneller als die PyArrays Variante)
Zunächst wird dann ein neuer 2D Vektor new_grid erstellt, auf den das grid kopiert wird.
Die Funktionsweise des Game of Life selbst ist dann wie bekannt. Auf das new_grid werden jeweils
die neuen Werte geschrieben während das grid selbst unverändert bleibt.
Eine Rückgabe kann erfolgen, da ein Vektor im Gegensatz zu einem Array in C++ zurückgegeben werden kann.
Die Rückgabe erfolgt nach Abschluss der vorgegebenen Iterationen.
*/
std::vector<std::vector<int>> gameOfLifeVektoren(std::vector<std::vector<int>> &grid, int iterations) {
    std::vector<std::vector<int>> new_grid;
    new_grid = grid;
    for (int i = 0; i < iterations; i++) {
        int N = grid.size();
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                new_grid[y][x] = 0;
                int z = neighbors(y, x, grid);
                if (z == 3) {
                    new_grid[y][x] = 1;
                    continue;
                }
                if ((grid[y][x] == 1) && (z == 2)) {
                    new_grid[y][x] = 1;
                }
            }
        }
        grid = new_grid;
    }
    return grid;
}


/*
Die Funktionalität der parallelen Funktion ist dieselbe wie die der seriellen Funktion.
Die Anwendung ist dabei disselbe wie in C++.
Eine Parallelisierung der Iterationen wäre nicht sinnvoll, für eine Folgegeneration
muss ja zunächst feststehen, aus welchen Werten diese ermittelt werden soll.
Diese findet dann sher komfortabel völlig automatisch statt.
*/


std::vector<std::vector<int>> gameOfLifeVParallel(std::vector<std::vector<int>>& grid, int iterations) {
    std::vector<std::vector<int>> new_grid;
    for (int i = 0; i < iterations; i++) {
        new_grid = grid;
        int N = grid.size();
        #pragma omp parallel for
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                new_grid[y][x] = 0;
                int z = neighbors(y, x, grid);
                if (z == 3) {
                    new_grid[y][x] = 1;
                    continue;
                }
                if ((grid[y][x] == 1) && (z == 2)) {
                    new_grid[y][x] = 1;
                }
            }
        }
        grid = new_grid;
    }
    return grid;
}

/*
Dieser Part gehört zu der PyBind11 Funktionalität und exportiert die Funktionen
für eine Benutzung aus Python heraus.
Eine Importierung in python kann dabei ganz standardmäßig mit
dem import Befehl vorgenommen werden.
z.B. mit: from GoLPybindsFunctions import gameOfLifeVektoren, gameOfLifeVParallel
*/
PYBIND11_MODULE(GoLPybindsFunctions, m) {
    m.def("gameOfLifeVektoren", &gameOfLifeV, "C");

    m.def("gameOfLifeVParallel", &gameOfLifePara, "C");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}