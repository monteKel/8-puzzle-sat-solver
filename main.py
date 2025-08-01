import streamlit as st
import pathlib
import numpy as np
import pandas as pd
from pysat.solvers import Glucose4
from random import randint
import time
import re
from collections import defaultdict
from PIL import Image

########################################################################################################################

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
   
def AtLeastAtMostOnePieceByCell(dict,solver,maxsteps=8,piecesNumber = 9,puzzleSize=3):
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

def AtLeastAtMostOnePieceByMatrix(dict,solver,maxsteps=8,piecesNumber = 9,puzzleSize=3):
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
            
def AtLeastAtMostOneActionByStep(dict,solver,maxsteps=8):
    for step in range(0,maxsteps,1):
        aux = [ActionValue(dict,step,"U"),ActionValue(dict,step,"D"),ActionValue(dict,step,"L"),ActionValue(dict,step,"R")]
        solver.add_clause(aux)
        for index in aux:
            for next in aux:
                if next > index:
                    solver.add_clause([(index * -1), (next * -1)])
    print("Fourth Step Done!!!\nClauses Counted:\n Each Step have to make One Action: Up, Down, Left Or Right\n In the last step, no actions are made\n\n")

def StateTransictionClauses(dict,solver,maxsteps=8,piecesNumber = 9,puzzleSize=3):
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

def InertiaClauses(dict,solver,maxsteps=8,piecesNumber = 9,puzzleSize=3):
    for step in range(0,maxsteps,1):
        for column in range(puzzleSize):
            for line in range (puzzleSize):
                for piece in range (1,piecesNumber):
                    aux = [(PossibilityValue(dict,step,line,column,piece) * -1), PossibilityValue(dict,step+1,line,column,0), PossibilityValue(dict,step+1,line,column,piece)]
                    solver.add_clause(aux)
    print("Sixth Step Done!!!\nClauses Counted:\n If piece X is in a position that doenst change with the step action,\nthen piece X will be in the same position on next Step\n\n")
                        
def CreateFinalTable(dict,solver,maxstep=8):
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

def CreateInitialState(dict,solver,aux):
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

def CreateStepMatrix(stringfysoluction):
    step_placeholder = st.empty()
    matrix_placeholder = st.empty()

    step_states = defaultdict(lambda: [[-1 for _ in range(3)] for _ in range(3)])
    actions = {}

    for literal in stringfysoluction:
        if "_A_" in literal:
            match = re.match(r"(\d+)_A_([UDLR])", literal)
            if match:
                step = int(match.group(1))
                direction = match.group(2)
                actions[step] = direction
        elif "_P_" in literal:
            match = re.match(r"(\d+)_P_(\d)_(\d)_(\d+)", literal)
            if match:
                step, i, j, piece = map(int, match.groups())
                step_states[step][i][j] = piece

    for step in sorted(step_states.keys()):
        matrix = step_states[step]
        direction = actions.get(step, None)

        if direction:
            dir_text = {"U": "↑ (Up)", "D": "↓ (Down)", "L": "← (Left)", "R": "→ (Right)"}.get(direction, direction)
            step_placeholder.markdown(f"### Passo {step}: {dir_text}")
        else:
            step_placeholder.markdown(f"### Passo {step}")

        df = pd.DataFrame(matrix)
        styled_html = df.style \
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
        matrix_placeholder.markdown(styled_html, unsafe_allow_html=True)
        time.sleep(0.9)

        # Checar se é a solução final
        if matrix == [[0,1,2],[3,4,5],[6,7,8]]:
            step_placeholder.markdown(f"### Solução final alcançada no passo {step}!")
            break


def GenerateRandomMatrix():
        matrix = np.zeros((3,3),dtype=int)
        piece = 0
        LineEmpty = 0
        ColumnEmpty = 0
        for line in range(3):
            for column in range(3):
                matrix[line][column] = piece
                piece+=1
        for QttMovements in range(100):
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

def RunSolver(matrix_input, max_limit=20):
    from pysat.solvers import Glucose4

    for steps in range(1, max_limit + 1):
        solver = Glucose4()
        MapDict = MapDictCreator(steps)
        AtLeastAtMostOnePieceByCell(MapDict,solver, steps)
        AtLeastAtMostOnePieceByMatrix(MapDict,solver, steps)
        AtLeastAtMostOneActionByStep(MapDict,solver, steps)
        StateTransictionClauses(MapDict,solver, steps)
        InertiaClauses(MapDict,solver, steps)
        CreateFinalTable(MapDict,solver, steps)
        CreateInitialState(MapDict,solver, matrix_input)

        if solver.solve():
            raw_solution = solver.get_model()
            string_solution = sorted(LiteralIndex(MapDict, x) for x in raw_solution if x > 0)
            return string_solution, steps, MapDict, raw_solution
    return None, None, None

if 'ValidMatrix' not in st.session_state:
    st.session_state.ValidMatrix = False
if 'PuzzleMatrix' not in st.session_state:
    st.session_state.PuzzleMatrix = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ], dtype=int)

numberSteps = 20



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
    EmptySize(2)
    EmptySize(2)


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


    if st.button("Resolver Matriz"):
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
                        st.write("Indices iguais :(")
                        st.session_state.ValidMatrix = not st.session_state.ValidMatrix
            if st.session_state.ValidMatrix:
                solution, steps, MapDict, rawSoluction = RunSolver(aux,numberSteps)
                st.write("Matriz Válida!! :)")
                if solution:
                    st.success(f"Solução encontrada com {steps} passo(s)!")
                    CreateStepMatrix(solution)
                else:
                    pass
            else:
                st.write("Error 404: Soluction Not Found")
                st.write("Generating a new random Matrix")
                dt = GenerateRandomMatrix()

with right:
    EmptySize(16)
    st.write("Envie uma imagem para resolver o puzzle de forma mais divertida :D")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    tiles_img = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        width, height = image.size
        new_width = (width // 3) * 3
        new_height = (height // 3) * 3
        image = image.resize((new_width, new_height))

        tile_w, tile_h = new_width // 3, new_height // 3

        tiles_img = {}

        for row in range(3):
            for col in range(3):
                left = col * tile_w
                upper = row * tile_h
                right = left + tile_w
                lower = upper + tile_h
                tile = image.crop((left, upper, right, lower))
                idx = row * 3 + col
                tiles_img[idx] = tile

with left:
    EmptySize(22)
    if st.button("Randomizar Matriz"):
        dt = GenerateRandomMatrix()
