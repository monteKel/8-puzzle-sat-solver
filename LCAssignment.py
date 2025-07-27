from pysat.solvers import Glucose4
import numpy as np
from random import randint

solver = Glucose4()

def MapDictCreator(maxsteps=8,piecesNumber=9,puzzleSize=3):
    PossibleActions = ["U","D","L","R"]
    mappingDict = {}
    reverseMappingDict = {}
    dictIndex = 1
    for step in range(0,maxsteps+1):
        for piece in range (0,piecesNumber,1):
            for line in range (0,puzzleSize,1):
                for column in range (0,puzzleSize,1):
                    mappingIndex = "{}_P_{}_{}_{}".format(step,line,column,piece)
                    mappingDict.update({mappingIndex: dictIndex})
                    dictIndex +=1
        if step != maxsteps:
            for action in range (4):
                mappingIndex = "{}_A_{}".format(step,PossibleActions[action])
                mappingDict.update({mappingIndex: dictIndex})
                dictIndex +=1
    print("First Step Done!!!\nMapping Dictionary Created!!!\n\n")
    return mappingDict

def PossibilityValue(dict,step,line,column,number):
    key = "{}_P_{}_{}_{}".format(step,line,column,number)
    if key in dict:
        aux = dict[key]
        return aux 
    else:
        print("The Coords number is not a avalible literal!!!")
        print("Number that is in problem focus:{}".format(key))

def LiteralIndex(dict, key):
    for index, literal in dict.items():
        if literal == key:
            return index
    return None

def ActionValue(dict,step,direction):
    key = "{}_A_{}".format(step,direction)
    if key in dict:
        aux = dict[key]
        return aux
    else:
        print("The Action numbers is not a avalible literal!!!")
        print("Number that is in problem focus:{}".format(key))
    
def DictPrint(dict):
    for index,key in enumerate(dict):
        print(f"Coordinate: {key}",f"Literal: {index+1}")
   
def AtLeastAtMostOnePieceByCell(dict,maxsteps=8,piecesNumber = 9,puzzleSize=3):
    for step in range (0,maxsteps+1,1):
        for line in range (puzzleSize):
            for column in range(puzzleSize):
                aux = []
                for piece in range(piecesNumber):
                    aux.append(PossibilityValue(dict,step,line,column,piece))
                solver.add_clause(aux)
                for index in aux:
                    for next in aux:
                        if next > index:
                            solver.add_clause([(index * -1), (next * -1)])
    print("Second Step Done!!!\nClauses Counted:\n Has to be one piece per cell: P1 or P2 or P3 etc...\n Has to be at most one piece per cell: ~(P1 & P2) = (~P1 v ~P2) etc...\n\n")

def AtLeastAtMostOnePieceByMatrix(dict,maxsteps=8,piecesNumber = 9,puzzleSize=3):
    for step in range (0,maxsteps+1,1):
        for piece in range(piecesNumber):
            aux = []
            for line in range (puzzleSize):
                for column in range(puzzleSize):
                    aux.append(PossibilityValue(dict,step,line,column,piece))
            solver.add_clause(aux)
            for index in aux:
                for next in aux:
                    if next > index:
                        solver.add_clause([(index * -1), (next * -1)])
    print("Third Step Done!!!\nClauses Counted:\n Has to be one piece per Matrix: P1 or P2 or P3 etc...\n Has to be at most one piece per Matrix: P1 and ~P2 and ~P3 etc...\n\n")
            
def AtLeastAtMostOneActionByStep(dict,maxsteps=8):
    for step in range(0,maxsteps,1):
        aux = [ActionValue(dict,step,"U"),ActionValue(dict,step,"D"),ActionValue(dict,step,"L"),ActionValue(dict,step,"R")]
        solver.add_clause(aux)
        for index in aux:
            for next in aux:
                if next > index:
                    solver.add_clause([(index * -1), (next * -1)])
    print("Fourth Step Done!!!\nClauses Counted:\n Each Step have to make One Action: Up, Down, Left Or Right\n In the last step, no actions are made\n\n")

def StateTransictionClauses(dict,maxsteps=8,piecesNumber = 9,puzzleSize=3):
    movement = ["U","D","L","R"]
    for move in movement:
        for step in range(0,maxsteps,1):
            for column in range(puzzleSize):
                for line in range (puzzleSize):
                    if move == "U" and line >= 1:
                        for piece in range(1,piecesNumber):
                            IfStatement=[(PossibilityValue(dict,step,line,column, 0) * -1) , (PossibilityValue(dict,step,line -1,column, piece) * -1), (ActionValue(dict,step,move) * -1)]
                            PieceSoStatement=[(PossibilityValue(dict,step+1,line,column, piece))]
                            VoidSoStatement=[(PossibilityValue(dict,step+1,line -1,column, 0))]
                            PieceClause = IfStatement + PieceSoStatement
                            VoidClause = IfStatement + VoidSoStatement
                            solver.add_clause(PieceClause)
                            solver.add_clause(VoidClause)
                    elif move == "D" and line <= 1:
                        for piece in range(1,piecesNumber):
                            IfStatement=[(PossibilityValue(dict,step,line,column, 0) * -1) , (PossibilityValue(dict,step,line +1,column, piece) * -1), (ActionValue(dict,step,move) * -1)]
                            PieceSoStatement=[(PossibilityValue(dict,step+1,line,column, piece))]
                            VoidSoStatement=[(PossibilityValue(dict,step+1,line +1,column, 0))]
                            PieceClause = IfStatement + PieceSoStatement
                            VoidClause = IfStatement + VoidSoStatement
                            solver.add_clause(PieceClause)
                            solver.add_clause(VoidClause)
                    elif move == "L" and column >= 1:
                        for piece in range(1,piecesNumber):
                            IfStatement=[(PossibilityValue(dict,step,line,column, 0) * -1) , (PossibilityValue(dict,step,line,column -1, piece) * -1), (ActionValue(dict,step,move) * -1)]
                            PieceSoStatement=[(PossibilityValue(dict,step+1,line,column, piece))]
                            VoidSoStatement=[(PossibilityValue(dict,step+1,line,column -1, 0))]
                            PieceClause = IfStatement + PieceSoStatement
                            VoidClause = IfStatement + VoidSoStatement
                            solver.add_clause(PieceClause)
                            solver.add_clause(VoidClause)
                    elif move == "R" and column <= 1:
                        for piece in range(1,piecesNumber):
                            IfStatement=[(PossibilityValue(dict,step,line,column, 0) * -1) , (PossibilityValue(dict,step,line,column +1, piece) * -1), (ActionValue(dict,step,move) * -1)]
                            PieceSoStatement=[(PossibilityValue(dict,step+1,line,column, piece))]
                            VoidSoStatement=[(PossibilityValue(dict,step+1,line,column +1, 0))]
                            PieceClause = IfStatement + PieceSoStatement
                            VoidClause = IfStatement + VoidSoStatement
                            solver.add_clause(PieceClause)
                            solver.add_clause(VoidClause)
                    if line == 0:
                        aux=[(PossibilityValue(dict,step,line,column,0) * -1),ActionValue(dict,step,"U") * -1]
                        solver.add_clause(aux)
                    if line == puzzleSize -1:
                        aux=[(PossibilityValue(dict,step,line,column,0) * -1),ActionValue(dict,step,"D") * -1]
                        solver.add_clause(aux)
                    if column == 0:
                        aux=[(PossibilityValue(dict,step,line,column,0) * -1),ActionValue(dict,step,"L") * -1]
                        solver.add_clause(aux)
                    if column == puzzleSize-1:
                        aux=[(PossibilityValue(dict,step,line,column,0) * -1),ActionValue(dict,step,"R") * -1]
                        solver.add_clause(aux)
    print("Fifth Step Done!!!\nClauses Counted:\n If piece piece X is in a position and piece The Action change places with X and Empty\nIn The Next step, they will have places changed \n\n")

def InertiaClauses(dict,maxsteps=8,piecesNumber = 9,puzzleSize=3):
    for step in range(0,maxsteps,1):
        for column in range(puzzleSize):
            for line in range (puzzleSize):
                for piece in range (1,piecesNumber):
                    aux = [(PossibilityValue(dict,step,line,column,piece) * -1), PossibilityValue(dict,step+1,line,column,0), PossibilityValue(dict,step+1,line,column,piece)]
                    solver.add_clause(aux)
    print("Sixth Step Done!!!\nClauses Counted:\n If piece X is in a position that doenst change with the step action,\nthen piece X will be in the same position on next Step\n\n")
                        
def CreateFinalTable(dict,maxstep=8):
    matrix = np.zeros((3,3))
    piece = 0
    for line in range(3):
        for column in range(3):
            matrix[line][column] = piece
            FinalState = PossibilityValue(dict,maxstep,line,column,piece)
            solver.add_clause([FinalState])
            piece += 1
    print("The Final Matrix has to looks like this:\n")
    print(matrix)

def CreateInitialState(dict):
    matrix = np.zeros((3,3))
    piece = 0
    LineEmpty = 0
    ColumnEmpty = 0
    for line in range(3):
        for column in range(3):
            matrix[line][column] = piece
            piece+=1
    for QttMovements in range(20):
        Movement = randint(0,3)
        if Movement == 0 and LineEmpty >= 1:
            aux = matrix[LineEmpty-1][ColumnEmpty]
            matrix[LineEmpty-1][ColumnEmpty] = matrix[LineEmpty][ColumnEmpty]
            matrix[LineEmpty][ColumnEmpty] = aux
            LineEmpty -= 1
        elif Movement == 1 and LineEmpty <= 1:
            aux = matrix[LineEmpty+1][ColumnEmpty]
            matrix[LineEmpty+1][ColumnEmpty] = matrix[LineEmpty][ColumnEmpty]
            matrix[LineEmpty][ColumnEmpty] = aux
            LineEmpty += 1
        elif Movement == 2 and ColumnEmpty >= 1:
            aux = matrix[LineEmpty][ColumnEmpty-1]
            matrix[LineEmpty][ColumnEmpty-1] = matrix[LineEmpty][ColumnEmpty]
            matrix[LineEmpty][ColumnEmpty] = aux
            ColumnEmpty -= 1
        elif Movement == 3 and ColumnEmpty <= 1:
            aux = matrix[LineEmpty][ColumnEmpty+1]
            matrix[LineEmpty][ColumnEmpty+1] = matrix[LineEmpty][ColumnEmpty]
            matrix[LineEmpty][ColumnEmpty] = aux
            ColumnEmpty += 1
    print("The Initial Matrix is:\n")
    for l in range (3):
        for c in range (3):
            aux = PossibilityValue(dict,0,l,c,int(matrix[l][c]))
            solver.add_clause([aux])

def CreateStepMatrix(dict,stringfysoluction,maxsteps=8,piecesNumber=9):
    step= -1
    Soluction=False
    SoluctionList=[0,1,2,3,4,5,6,7,8]
    print("\n")
    for index in range(len(stringfysoluction)):
        action = stringfysoluction[index][2]
        if Soluction == False:
            if action == "A":
                aux=[]
                step+=1
                print("\nStep Number {}".format(step))
                if stringfysoluction[index][4] == "U":
                    print("Action that matrix will make: Upwards")
                elif stringfysoluction[index][4] == "D":
                    print("Action that matrix will make: Downwards")
                elif stringfysoluction[index][4] == "L":
                    print("Action that matrix will make: Leftwards")
                else:
                    print("Action that matrix will make: Rightwards")
            else:
                aux.append(int(stringfysoluction[index][8]))
                if len(aux) >= 9:
                    if aux == SoluctionList:
                        print("Were finally in the last step, or if you wanna say, the final matrix :D\n")
                        print("_____________\n| {} | {} | {} |".format(aux[0],aux[1],aux[2]))
                        print("| {} | {} | {} |".format(aux[3],aux[4],aux[5]))
                        print("| {} | {} | {} |\n_____________\n\n\n".format(aux[6],aux[7],aux[8]))
                        Soluction = True
                    else:
                        print("_____________\n| {} | {} | {} |".format(aux[0],aux[1],aux[2]))
                        print("| {} | {} | {} |".format(aux[3],aux[4],aux[5]))
                        print("| {} | {} | {} |\n_____________\n\n\n".format(aux[6],aux[7],aux[8]))
                        aux=[]

        

def main():
    MaxSteps = 9
    MapDict = MapDictCreator(MaxSteps)
    AtLeastAtMostOnePieceByCell(MapDict,MaxSteps)
    AtLeastAtMostOnePieceByMatrix(MapDict,MaxSteps)
    AtLeastAtMostOneActionByStep(MapDict,MaxSteps)
    StateTransictionClauses(MapDict,MaxSteps)
    InertiaClauses(MapDict,MaxSteps)
    CreateInitialState(MapDict)
    CreateFinalTable(MapDict,MaxSteps)

    if solver.solve():
        print("Valid Soluction")
        rawSoluction = solver.get_model()
        StringfySoluction = []
        for index in range(len(rawSoluction)):
            if rawSoluction[index] > 0:
                StringfySoluction.append(LiteralIndex(MapDict,rawSoluction[index]))
        StringfySoluction.sort()
        CreateStepMatrix(dict,StringfySoluction,MaxSteps)
    else:
        print("The Soluction is not possible")




main()