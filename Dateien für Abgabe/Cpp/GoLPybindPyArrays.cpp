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
Bekannte Hilfsfunktion
*/
int trueValueA(int x, int N) {
    return (x + N) % N;
}

/*
Die Funktion nimmt eine Position im Array über die x,y Koordinaten,
sowie das Array selbst als Referenz-Parameter an.
Da hier mit eindimensionalen Pointer Arrays gearbeitet wird, muss hier über
die vom PyArray bereitgestellten Buffer das zweidimenstionale grid auf ein
eindimensionales Pointer Array reduziert werden.
Um dann im eindimensionalen Arrays den richtigen Index zu finden, muss der
Parameter der zweiten Dimension (y) mit der Breite(=Höhe in unserem Fall)
des Grids verrechnet werden. Anschließend kann der Index gefunden werden, indem der x Wert,
also der Index der 1. Dimension auf den angepassten Wert der zweiten Dimension
addiert wird.
Von dieser Komplexität abgesehen, funktioniert die Funktion aber wie schon in der
C++ Implementation.
*/
int neighborsA(int y, int x, py::array_t<int> &grid) {
    py::buffer_info buf1 = grid.request();
    int* ptr1 = (int*)buf1.ptr; //eindimensionales Pointer Array
    int N = buf1.shape[0];
    int count = 0;
    int range[] = { -1, 0, 1 };
    for (int v : range) {
        for (int w : range) {
            if (w || v) {
                count += ptr1[trueValueA(y + v, N)* N + trueValueA(x + w, N)];
            }
        }
    }
    return count;
}

/*
Diese Funktion führt ein Game of Life auf einem als Parameter gegebenen (2D-)PyArray aus.
Zunächst wird dafür das PyArrays "kopiert" insofern, dass ein zweites 2D Array, result,
mit denselben Maßen erstellt wird.
Da hier mit eindimensionalen Pointer Arrays gearbeitet wird, müssen hier
anschließend beide Arrays von einem zweidimenstionalen grid auf ein
eindimensionales Pointer Array reduziert werden.
Um dann im eindimensionalen Arrays den richtigen Index zu finden, muss der
Parameter der zweiten Dimension (y) mit der Breite (hier gleich der Höhe)
des Grids verrechnet werden. Anschließend kann der Index gefunden werden, indem der x Wert,
also der Index der 1. Dimension auf den angepassten Wert der zweiten Dimension
addiert wird.
Von dieser Komplexität abgesehen, funktioniert die Funktion innerhalb der Schleifen
aber wie schon in der C++ Implementation.
Am Ende kann dann das Ergebnis (result) zurückgegeben werden.
Damit dieses in Python besser verwendet werden kann,
wird es allerdings vorher wieder in die ursprünglichen 2D Maße
überführt.
*/
py::array_t<int> gameOfLifePyArrays(py::array_t<int> &grid) {
    py::buffer_info buf1 = grid.request();
    auto result = py::array_t<int>(buf1.size);
    py::buffer_info buf2 = result.request();
    //Reduktion der beiden Pyarrays auf 1D Pointer-Array
    int *ptr1 = (int*)buf1.ptr;
    int *ptr2 = (int*)buf2.ptr;
    int N = buf1.shape[0];
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            ptr2[y*N + x] = 0;
            int z = neighborsA(y, x, grid);
            if (z == 3) {
                ptr2[y * N + x] = 1;
                continue;
            }
            if ((ptr1[y * N + x] == 1) && (z == 2)) {
                ptr2[y * N + x] = 1;
            }
        }
    }
    //Resizing des ErgebnisArrays auf ein 2D Array
    result.resize({N,N});
    return result;
}

/*
Dieser Part gehört zu der PyBind11 Funktionalität und exportiert die Funktionen
für eine Benutzung aus Python heraus.
Eine Importierung in python kann dabei ganz standardmäßig mit
dem import Befehl vorgenommen werden.
z.B. mit: from GoLPybindsFunctions import gameOfLifePyArrays
*/
PYBIND11_MODULE(GoLPybindsFunctions, m) {
    m.def("gameOfLifePyArrays", &gameOfLifePyArrays, "C");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}