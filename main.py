import streamlit as st
import pathlib
import numpy as np
import pandas as pd
from pysat.solvers import Glucose4
from random import randint

########################################################################################################################
solver = Glucose4()

def MapDictCreator(maxsteps=8,piecesNumber=9,puzzleSize=3):
    PossibleActions = ["U","D","L","R"]
    mappingDict = {}
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

def CreateInitialState(dict,aux):
    number = PossibilityValue(dict,0,0,0,aux[0])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,0,1,aux[1])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,0,2,aux[2])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,1,0,aux[3])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,1,1,aux[4])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,1,2,aux[5])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,2,0,aux[6])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,2,1,aux[7])
    solver.add_clause([number])
    number = PossibilityValue(dict,0,2,2,aux[8])
    solver.add_clause([number])

def CreateStepMatrix(dict,stringfysoluction,maxsteps=8,piecesNumber=9):
    step= -1
    Soluction=False
    SoluctionList=[0,1,2,3,4,5,6,7,8]
    st.write("")
    st.write("")
    for index in range(len(stringfysoluction)):
        action = stringfysoluction[index][2]
        if Soluction == False:
            if action == "A":
                aux=[]
                step+=1
                st.write("\nStep Number {}".format(step))
                if stringfysoluction[index][4] == "U":
                    st.write("Action that matrix will make: Upwards")
                elif stringfysoluction[index][4] == "D":
                    st.write("Action that matrix will make: Downwards")
                elif stringfysoluction[index][4] == "L":
                    st.write("Action that matrix will make: Leftwards")
                else:
                    st.write("Action that matrix will make: Rightwards")
            else:
                aux.append(int(stringfysoluction[index][8]))
                if len(aux) >= 9:
                    if aux == SoluctionList:
                        st.write("Were finally in the last step, or if you wanna say, the final matrix :D\n")
                        current_df_for_display = pd.DataFrame({" ":[aux[0],aux[3],aux[6]],"  ":[aux[1],aux[4],aux[7]],"   ":[aux[2],aux[5],aux[8]]})
                        styled_html = current_df_for_display.style \
                        .hide(axis="index") \
                        .hide(axis="columns") \
                        .set_table_styles([
                            {'selector': '', 'props': [
                                ('border-collapse', 'collapse'),
                                ('width', 'auto !important'),
                                ('margin-left', 'auto'),
                                ('margin-right', 'auto'),
                                ('font-size', '2.5em'),
                                ('text-align', 'center')
                            ]},
                            {'selector': 'td', 'props': [
                                ('border', '2px solid #6C757D'),
                                ('padding', '10px'),
                                ('width', '70px'),
                                ('height', '70px')
                            ]},
                            {'selector': 'td:empty', 'props': [
                                ('background-color', '#444444')
                            ]}
                        ]) \
                        .to_html()
                        st.markdown(styled_html, unsafe_allow_html=True)
                        Soluction = True
                    else:
                        current_df_for_display = pd.DataFrame({" ":[aux[0],aux[3],aux[6]],"  ":[aux[1],aux[4],aux[7]],"   ":[aux[2],aux[5],aux[8]]})
                        styled_html = current_df_for_display.style \
                        .hide(axis="index") \
                        .hide(axis="columns") \
                        .set_table_styles([
                            {'selector': '', 'props': [
                                ('border-collapse', 'collapse'),
                                ('width', 'auto !important'),
                                ('margin-left', 'auto'),
                                ('margin-right', 'auto'),
                                ('font-size', '2.5em'),
                                ('text-align', 'center')
                            ]},
                            {'selector': 'td', 'props': [
                                ('border', '2px solid #6C757D'),
                                ('padding', '10px'),
                                ('width', '70px'),
                                ('height', '70px')
                            ]},
                            {'selector': 'td:empty', 'props': [
                                ('background-color', '#444444')
                            ]}
                        ]) \
                        .to_html()
                        st.markdown(styled_html, unsafe_allow_html=True)
                        aux=[]

def GenerateRandomMatrix():
        matrix = np.zeros((3,3),dtype=int)
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
        st.session_state.PuzzleMatrix = matrix
        st.session_state.ValidMatrix = True
        return pd.DataFrame(st.session_state.PuzzleMatrix)

if 'ValidMatrix' not in st.session_state:
    st.session_state.ValidMatrix = False
if 'PuzzleMatrix' not in st.session_state:
    st.session_state.PuzzleMatrix = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ], dtype=int)
MaxSteps = 8
MapDict = MapDictCreator(MaxSteps)
AtLeastAtMostOnePieceByCell(MapDict,MaxSteps)
AtLeastAtMostOnePieceByMatrix(MapDict,MaxSteps)
AtLeastAtMostOneActionByStep(MapDict,MaxSteps)
StateTransictionClauses(MapDict,MaxSteps)
InertiaClauses(MapDict,MaxSteps)
CreateFinalTable(MapDict,MaxSteps)




with open("style.css") as f:
    st.markdown(f"<style>{f.read()}<style>",unsafe_allow_html=True)

left,center,right = st.columns([1,2,1])
df = pd.DataFrame({" ":[0,3,6],"  ":[1,4,7],"   ":[2,5,8]})

def EmptySize(size):
    for index in range(size):
        st.write("")

def LoadCss(path):
    with open(path) as f:
        st.html(f"<style>{f.read()}<style>")
cssPath = pathlib.Path("/LPCAssignment/style.css")



with center:
    st.title("8-Puzzle Solver",)
    st.subheader("with pysat glucose solver")
    EmptySize(3)


    for line in range(3):
        columns = st.columns(3)
        for column in range(3):
            with columns[column]:

                cell_class = "grid-cell"
                if st.session_state.PuzzleMatrix[line, column] == 0:
                    cell_class += " empty-cell"
                st.markdown(f'<div class="{cell_class}">', unsafe_allow_html=True)

                def update_matrix_cell(row, col):
                    st.session_state.PuzzleMatrix[row, col] = int(st.session_state[f"num_input_{row}_{col}"])

                st.number_input(
                    label="",
                    min_value=0,
                    max_value=8,
                    value=int(st.session_state.PuzzleMatrix[line, column]),
                    key=f"num_input_{line}_{column}",
                    on_change=update_matrix_cell,
                    args=(line, column)
                )
                
                st.markdown('</div>', unsafe_allow_html=True)

    EmptySize(2)
    current_df_for_display = pd.DataFrame(st.session_state.PuzzleMatrix)
    styled_html = current_df_for_display.style \
        .hide(axis="index") \
        .hide(axis="columns") \
        .set_table_styles([
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('width', 'auto !important'),
                ('margin-left', 'auto'),
                ('margin-right', 'auto'),
                ('font-size', '2.5em'), # Ajuste o tamanho da fonte para a exibição
                ('text-align', 'center')
            ]},
            {'selector': 'td', 'props': [
                ('border', '2px solid #6C757D'), # Borda para as células
                ('padding', '10px'),
                ('width', '70px'), # Largura/Altura da célula da tabela HTML
                ('height', '70px')
            ]},
            {'selector': 'td:empty', 'props': [ # Estilo para a célula vazia (0)
                ('background-color', '#444444')
            ]}
        ]) \
        .to_html()
    st.markdown(styled_html, unsafe_allow_html=True)


    if st.button("Solve This Matrix"):
        st.session_state.ValidMatrix = True
        with center:
            aux=[]
            selectedMatrix = st.session_state.PuzzleMatrix
            for line in range(3):
                for column in range(3):
                    aux.append(int(selectedMatrix[line][column]))
            for i in range(len(aux)):
                for j in range(len(aux)):
                    if i != j and aux[i] == aux[j] and st.session_state.ValidMatrix:
                        st.write("This matrix cant be done: Equal Indexes")
                        st.session_state.ValidMatrix = not st.session_state.ValidMatrix
            if st.session_state.ValidMatrix:
                CreateInitialState(MapDict,aux)
                st.write("This is a valid matrix :)")
                if solver.solve():
                    st.write("Searching soluction with:")
                    st.write("Default steps quantity: 8 Steps")
                    rawSoluction = solver.get_model()
                    StringfySoluction = []
                    for index in range(len(rawSoluction)):
                        if rawSoluction[index] > 0:
                            StringfySoluction.append(LiteralIndex(MapDict,rawSoluction[index]))
                    StringfySoluction.sort()
                    CreateStepMatrix(dict,StringfySoluction,MaxSteps)
                else:
                    st.write("Error 404: Soluction Not Found")
                    st.write("Generating a new random Matrix")
                    dt = GenerateRandomMatrix()


with right:
    EmptySize(22)

with left:
    EmptySize(22)
    if st.button("Randomize Matrix"):
        dt = GenerateRandomMatrix()
    st.write("If you randomize")
    st.write("please click on a index")
    st.write("erase and press enter")
    st.write("im tryin to fix this soon :)")
