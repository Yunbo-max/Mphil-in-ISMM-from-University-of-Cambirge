use io;

/* Read instance data */
function input() {
    local usage = "Usage: localsolver assembly_line_balancing.lsp "
            + "inFileName=inputFile [lsTimeLimit=timeLimit] [solFileName=solFile]";

    if(inFileName == nil) throw usage;
    local inFile = io.openRead(inFileName);

    inFile.readln();
    
    // Read number of tasks
    nbTasks = inFile.readInt();
    maxNbStations = nbTasks;
    inFile.readln();

    // Read the cycle time limit
    cycleTime = inFile.readInt();
    for [i in 0..4] inFile.readln();

    // Read the processing times
    for [t in 0..nbTasks-1]
        processingTime[inFile.readInt() - 1] = inFile.readInt();
    inFile.readln();

    // Read the successors' relations
    successors[0..nbTasks-1] = {};
    local line = inFile.readln().split(",");
    while(line.count() > 1) {
        local predecessor = toInt(line[0]) - 1;
        local successor = toInt(line[1]) - 1;
        successors[predecessor].add(successor);
        line = inFile.readln().split(",");
    }
    inFile.close();
}

/* Declare the optimization model */
function model() {
    // Decision variables: station[s] is the set of tasks assigned to station s
    station[s in 0..maxNbStations-1] <- set(nbTasks);
    constraint partition[s in 0..maxNbStations-1](station[s]);

    // Objective: nbUsedStations is the total number of used stations
    nbUsedStations <- sum[s in 0..maxNbStations-1](count(station[s]) > 0);

    // All stations must respect the cycleTime constraint
    timeInStation[s in 0..maxNbStations-1] <- sum(station[s], i => processingTime[i]);
    for [s in 0..maxNbStations-1]
        constraint timeInStation[s] <= cycleTime;

    // The stations must respect the succession's order of the tasks
    taskStation[i in 0..nbTasks-1] <- find(station, i);
    for [i in 0..nbTasks-1][j in successors[i]]
        constraint taskStation[i] <= taskStation[j];

    // Minimization of the number of active stations
    minimize nbUsedStations;
}

/* Parametrize the solver */
function param() {
    if (lsTimeLimit == nil) lsTimeLimit = 20;
}

/* Write the solution in a file following the format:
* - 1st line: value of the objective
* - 2nd line: number of tasks
* - following lines: task's number, station's number */
function output() {
    if(solFileName == nil) return;
    local solFile = io.openWrite(solFileName);
    solFile.println(nbUsedStations.value);
    solFile.println(nbTasks);
    for [i in 0..nbTasks-1]
        solFile.println(i + 1, ",", taskStation[i].value + 1);
}
