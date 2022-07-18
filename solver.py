import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            print("cost of dfs: ", cost_state(temp))
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    beginBox = PosOfBoxes(gameState)# Gán vị trí bắt đầu của Boxes vào biến beginBox
    beginPlayer = PosOfPlayer(gameState)#Gán vị trí bắt đầu của player vào biến beginPlayer
    startState = (beginPlayer, beginBox)  # Không gian trạng thái của bản đồ, lưu trữ vị trí bắt đầu của Boxes và player
    frontier = collections.deque([[startState]])  # queue store state, tạo hàng đợi queue frontier, node đầu tiên là vị trí bắt đầu của người chơi
    actions = collections.deque([[0]])  # queue store actions,tạo hàng đợi queue actions, ta xem vị trí ban đầu là trạng thái 0
    exploredSet = set() #tập các điểm đã đi qua, dùng set để lưu trữ và tránh trùng lặp
    temp=[] #lưu trữ những đường đi đến được đích 
    while frontier:  # lấy các phần tử của frontier cho đến khi rỗng
        node = frontier.popleft() # Lấy ra list ở đầu frontier queue lưu tập các states
        node_action = actions.popleft() # Lấy ra list ở đầu action queue lưu tập các actions
        if isEndState(node[-1][-1]): # Nếu node cuối cùng giống với goal thì ta lấy cộng vào temp đường đi đó
            temp += node_action[1:]  #Lưu actions vào temp, bỏ qua phần tử 0 của action
            print("cost of bfs: ", cost_state(temp)) #in chi phí
            break;
        if node[-1] not in exploredSet:# kiểm tra xem state mới nhất ở cuối node list đã đuợc explore chưa
            exploredSet.add(node[-1])#thêm node[-1] vào đầu exploredSet
            for action in legalActions(node[-1][0], node[-1][1]):# tìm kiếm các action mà ta có thể thực hiện dựa trên các vị trí lấy từ state
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)#cập nhật tọa độ của hộp và người chơi từ action
                if isFailed(newPosBox):#  Nếu các box bị mắc kẹt ở vị trí không hợp lệ, bỏ qua và chuyển sang action khác
                    continue;
                frontier.append(node + [(newPosPlayer, newPosBox)])#thêm tập states cũ cùng state mới vào frontier
                actions.append(node_action + [action[-1]]) #Thêm tập actions cũ cùng action mới vào frontier
    return temp # Trả về list hành động để đến được đích

def cost(actions):
    """A cost function"""
    return actions.count('l')+actions.count('r')+actions.count('u')+actions.count('d') #trả về tổng chi phí thông qua việc đếm các action l,r,u,d
def cost_state(actions):
    return len([x for x in actions ]) #trả về tổng chi phí thông qua việc đếm các action cả hoa cả thường
def uniformCostSearch(gameState):
    beginBox = PosOfBoxes(gameState) # Gán vị trí bắt đầu của Boxes vào biến beginBox
    beginPlayer = PosOfPlayer(gameState) #Gán vị trí bắt đầu của player vào biến beginPlayer
    startState = (beginPlayer, beginBox) # Không gian trạng thái của bản đồ, lưu trữ vị trí bắt đầu của Boxes và player
    frontier = PriorityQueue() # queue store states, tạo hàng đợi queue frontier, node đầu tiên là vị trí bắt đầu của người chơi
    frontier.push([startState], 0) # Bắt đầu frontier bằng biến startState, trọng số bằng 0
    exploredSet = set() #tập các điểm đã đi qua, dùng set để lưu trữ và tránh trùng lặp
    actions = PriorityQueue() # queue store actions, ta xem vị trí ban đầu là trạng thái 0
    actions.push([0], 0) # Bắt đầu frontier mà không có hành động, trọng số bằng 0
    temp = [] #lưu trữ những đường đi đến được đích 
    ### Implement uniform cost search here
    while frontier:  # lấy các phần tử của frontier cho đến khi rỗng
        node = frontier.pop() #Lấy ra list ở đầu frontier queue có trọng số nhỏ nhất, rồi lưu vào tập state
        node_action = actions.pop()  #Lấy ra list ở đầu action queue có trọng số nhỏ nhất, rồi lưu vào tập actions
        if isEndState(node[-1][-1]): #Kiểm tra nếu tìm ra được kết quả (boxes đã ở đích)
            temp += node_action[1:] #Lưu actions vào temp, bỏ qua phần tử 0 của action
            print("cost of ucs: ", cost(temp)) #in chi phí
            break
        if node[-1] not in exploredSet: # kiểm tra xem state mới nhất ở cuối node list đã đuợc explore chưa
            exploredSet.add(node[-1]) #thêm node[-1] vào đầu exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # tìm kiếm các action mà ta có thể thực hiện dựa trên các vị trí lấy từ state
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)#cập nhật tọa độ của hộp và người chơi từ action
                if isFailed(newPosBox):  #  Nếu các box bị mắc kẹt ở vị trí không hợp lệ, bỏ qua và chuyển sang action khác
                    continue
                #temp = node_action + [action[-1]]                                                    
                frontier.push(node + [(newPosPlayer, newPosBox)],cost(node_action + [action[-1]]))  #thêm tọa độ của hộp và nguyời chơi với chi phí tốt nhất vào frontier
                actions.push(node_action + [action[-1]],cost(node_action + [action[-1]])) #thêm hành động đã hoàn thành với chi phí tốt vào frontier
    return temp # Trả về list hành động để đến được đích                                    

def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method, indexlevel):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Level %i, Runtime of %s: %.2f second.' %(indexlevel, method, time_end-time_start))
    print(result)
    return result
