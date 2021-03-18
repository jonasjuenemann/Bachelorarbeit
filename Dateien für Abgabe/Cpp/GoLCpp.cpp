#include <iostream>
//omp.h muss nur importiert werden, wenn parallelisiert werden soll.
#include <omp.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <random>

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

const int BOARDSIZE = 2048; //Eine globalen Konstante, um Array-Probleme zu vermeiden.

// ----------------------------------------------------------------------------
// Hilfsfunktion um sich das Grid in erkennbarer Form anzeigen zu lassen.
// Die Korrektheit des Game of Life wurde bei der C++ Ausführung manuell geprüft,
// dafür ist diese Funktion sehr hilfreich.

void printGrid(int(&grid)[BOARDSIZE][BOARDSIZE]) {
    for (int i = 0; i < BOARDSIZE; i++)
    {
        for (int j = 0; j < BOARDSIZE; j++)
        {
            std::cout << " " << grid[i][j] << " ";
        }
        std::cout << "\n";
    }
}

/*
"""
Die Funktion nimmt eine Koordinate des Arrays als Parameter entgegen.
Liegt die Koordinate innerhalb der Breite des Arrays wird diese ohne
Veränderung zurückgegeben.
Sollte die Koordinate außerhalb der Breite und damit des Array Bereichs liegen,
wird diese durch die Modulo Funktionalität auf den Anfang des Arrays gesetzt.
Die Funktion gibt die ermittelte Koordinate zurück.
"""
*/
int trueValue(int x) {
    return (x + BOARDSIZE) % BOARDSIZE;
}
/*
Die Funktion nimmt eine Position im Array über die x,y Koordinaten,
sowie das Array selbst als Referenz-Parameter an.
Es wird zunächst ein Counter initialisiert, der auf 0 gestellt wird.
Anschließend wird in einer doppelten for-Schleife über die Nachbarn der Zelle
iteriert. Dies geschieht hier auf etwas andere Weise
als in der Python Implementation, die FUnktionalität ist aber dieselbe.
Hierbei wird der "wahre Wert" des Nachbarn zuvor bestimmt, um die Randfälle
abzudecken, an denen eine solche Abfrage sonst außerhalb Bereichs läge.
Die Funktion gibt eine Addition der Werte, die diese Nachbarn besitzen, zurück.
*/
int neighbors(int y, int x, int (&grid)[BOARDSIZE][BOARDSIZE]) {
    int count = 0;
    // Werte für die direkten Nachbarn der Zelle. (x-1 -> x+1, y-1 -> x+1)
    int range[] = { -1, 0, 1 };
    for (int v : range) {
        for (int w : range) {
            // Der Wert des betrachteten Punkts (wenn w und v 0 sind),
            // wird hier ausgelassen, da kein Nachbar.
            if (w || v) {
                //Ansonsten wird der Counter erhöht
                count += grid[trueValue(y + w)][trueValue(x + v)];
            }
        }
    }
    return count;
}

/*
Diese Funktion führt ein Game of Life auf einem als Parameter gegebenen (2D-)Array aus.
Das 2D-Array wird hier als Referenz übergeben. (Wichtig!)
Dann wird zunächst eine exakte Kopie des Arrays erstellt und die Breite des Arrays ermittelt.
Anschließend wird mit einer doppelten for-Schleife über das Array iteriert.
Dabei wird für jede Zelle betrachtet, wie viele lebendige Nachbarn diese Zelle hat.
Bei drei Nachbarn wird der Punkt mit einer 1 markiert (bleibt lebendig), bei zwei bleibt die Zelle Lebendig,
wenn diese schon zuvor lebendig ist. In jedem anderen Fall stirbt die Zelle. (und wird mit einer 0 markiert)
Die neuen Werte werden dabei in dem übergebenen Array gespeichert.
Die Anfang erstellte Kopie des Arrays wird über die Funktion hinweg nicht verändert.
Eine Rückgabe erfolgt nicht, ein 2D Array kann in C++ auch nicht zurückgegeben werden.
Das EingabeArray wurde aber im Speicher verändert und die Ergebnisse können daher genutzt werden.
*/
void gameOfLife(int(&grid)[BOARDSIZE][BOARDSIZE]) {
    static int original[BOARDSIZE][BOARDSIZE]; //Keine Memory Allocation im Stack durch static
    //Kopie des Übergebenen Arrays auf das neu erstellte Array.
    std::copy(&grid[0][0], &grid[0][0] + BOARDSIZE * BOARDSIZE, &original[0][0]);
    int N = BOARDSIZE;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            grid[y][x] = 0;
            int z = neighbors(y, x, original);
            if (z == 3) {
                grid[y][x] = 1;
                continue;
            }
            if ((original[y][x] == 1) && (z == 2)) {
                grid[y][x] = 1;
            }
        }
    }
}

/*
Diese Funktionalität der Parallelen Funktion ist dieselbe wie die der seriellen.
Der entscheidende Unterschied ist, dass hier die äußere for-Schleife mit der Direktive
#pragma omp parallel for ausgestattet wurde. Dies weist OpenMp an, hier zu parallelisieren.
Diese findet dann sher komfortabel völlig automatisch statt.
*/
void gameOfLifeParallel(int(&grid)[BOARDSIZE][BOARDSIZE]) {
    static int original[BOARDSIZE][BOARDSIZE];
    std::copy(&grid[0][0], &grid[0][0] + BOARDSIZE * BOARDSIZE, &original[0][0]);
    int N = BOARDSIZE;
    //nur aeußere Parallelisierung macht Sinn, eine Parallelisierung der inneren Schleife
    //bringt keinen Speedup, nicht genug Berechnung fuer die Parallelisierung vorliegt.
    #pragma omp parallel for
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            grid[y][x] = 0;
            int z = neighbors(y, x, original);
            if (z == 3) {
                grid[y][x] = 1;
                continue;
            }
            if ((original[y][x] == 1) && (z == 2)) {
                grid[y][x] = 1;
            }
        }
    }
}



int main() {
    //Anzahl der jeweils durchgeführten Iterationen
    int numIterations = 20;
    //Anzahl der Durchführungen über die ein Durchschnitt gebildet wird.
    int numDurchf = 100;
    std::cout << "Starting:" << "\n";
    //Die Messung der Performance erfolgt über das Clock Werkzeug, das in C++
    //automatisch integriert ist.
    clock_t startTime = clock();

    //Hier wird ein Nummerngenerator mit einer Gleichverteilung zwischen 0 und 1 erstellt.
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    //Erstellen eines InitialArrays
    static int grid[BOARDSIZE][BOARDSIZE];
    for (int y = 0; y < BOARDSIZE; y++) {
        for (int x = 0; x < BOARDSIZE; x++) {
            double number = distribution(generator);
            if (number > 0.75) {
                grid[y][x] = 1;
                continue;
            }
            else {
                grid[y][x] = 0;
            }
        }
    }

    // Ausführen des GoL über alle Dürchführungen
    // mit jeweils der zuvor definierten Anzahl an Iterationen.
    for (int x = 0; x < numDurchf; x++) {
        for (int i = 0; i < numIterations; i++) {
            gameOfLife(grid);
        }
    }
    clock_t endTime = clock();
    //Messen der Zeit
    clock_t clockTicksTaken = endTime - startTime;
    //Umrechnen der Zeit in Sekunden und Bilden eines Durchschnitts
    //Über alle Durchführungen.
    double timeInSeconds = (clockTicksTaken / (double)CLOCKS_PER_SEC)/ numDurchf;

    std::cout << "Time to execute in Serial was: " << timeInSeconds << "\n";

    return 0;
};
