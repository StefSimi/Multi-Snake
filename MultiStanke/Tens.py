import time
import pygame
import random
import neat
import Astar
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from collections import deque
import copy


def build_model(input_shape, output_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model((state_size,), action_size)
        self.target_model = build_model((state_size,), action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

window_x = 720
window_y = 480

# Okay so I need a game class that contains the list of snakes and the map.
# Because I need to run like 1000 games at the same time or sth

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

pygame.init()

pygame.display.set_caption('Snake')
game_window = pygame.display.set_mode((window_x, window_y))

fps = pygame.time.Clock()

#map = [[0 for _ in range(48)] for _ in range(72)]


def get_submatrix(matrix, X, P):
    rows = len(matrix)
    cols = len(matrix[0])

    # Center coordinates
    cx, cy = X

    # Define the bounds of the submatrix
    start_row = cx - P
    end_row = cx + P + 1
    start_col = cy - P
    end_col = cy + P + 1

    # Initialize the submatrix with 1s
    submatrix = [[1 for _ in range(2 * P + 1)] for _ in range(2 * P + 1)]

    # Copy the values from the original matrix to the submatrix
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            if 0 <= i < rows and 0 <= j < cols:
                submatrix[i - start_row][j - start_col] = matrix[i][j]
    '''transposed_matrix = [list(row) for row in zip(*submatrix)]
    for row in transposed_matrix:
        print(row)'''

    return submatrix
    '''N = len(matrix)  # Number of rows
    M = len(matrix[0])  # Number of columns
    center_i, center_j = X  # Coordinates of the center point X

    submatrix_size = 2 * P + 1

    # Initialize the submatrix
    submatrix = [[0] * submatrix_size for _ in range(submatrix_size)]
    # for i in range(center_i-P,center_i+P+1):
    # for j in range(center_j-P,center_j+P+1):
    # submatrix[i-center_i+P][j-center_j+P]=matrix[i][j]
    # print(i,j)
    # print(submatrix)
    # for row in submatrix:
    # print(row)

    # Define the size of the submatrix

    for i in range(submatrix_size):
        for j in range(submatrix_size):
            # Calculate the original matrix coordinates with wrapping
            orig_i = (center_i - P + i) % N
            orig_j = (center_j - P + j) % M
            submatrix[i][j] = matrix[orig_i][orig_j]
    transposed_matrix = [list(row) for row in zip(*submatrix)]
    # for row in transposed_matrix:
    # print(row)
    return submatrix'''


def clamp(x, min, max):
    if (x < min):
        return min
    if (x > max):
        return max
    return x


class Snake:


    def __init__(self, color, initial_position, speed, P, PV, RInit, playerControlled):
        self.body = [initial_position]
        self.body.insert(len(self.body) - 1, list(self.body[-1]))
        self.body.insert(len(self.body) - 1, list(self.body[-1]))
        self.direction = pygame.K_RIGHT
        self.change_to = self.direction
        self.color = color
        self.grow = False
        self.speed = 60 - speed + 1
        self.P = P
        self.PV = PV
        self.RInit = RInit
        self.playerControlled = playerControlled

        self.foodPath = []
        self.poisonedFoodPath = []
        self.frameCounter = 0
        self.pCounter = 0
        self.pvCounter = 0
        self.output = [1, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        # [1, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 4, 3, 10, 6, 0, 0]
        self.fitness = 0



    def getOutput(self, sightMatrix):
        self.output = []

        # WALL
        flag = 0
        for i in range(self.RInit, 0, -1):
            if (sightMatrix[i][self.RInit] == 1):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit, 0, -1):
            if (sightMatrix[self.RInit][i] == 1):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit + 1, self.RInit * 2 + 1):
            if (sightMatrix[i][self.RInit] == 1):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit + 1, self.RInit * 2 + 1):
            if (sightMatrix[self.RInit][i] == 1):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)

        # FOOD
        flag = 0
        for i in range(self.RInit, 0, -1):
            if (sightMatrix[i][self.RInit] == 3):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit, 0, -1):
            if (sightMatrix[self.RInit][i] == 3):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit + 1, self.RInit * 2 + 1):
            if (sightMatrix[i][self.RInit] == 3):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit + 1, self.RInit * 2 + 1):
            if (sightMatrix[self.RInit][i] == 3):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)

        # POISON
        flag = 0
        for i in range(self.RInit, 0, -1):
            if (sightMatrix[i][self.RInit] == 4):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit, 0, -1):
            if (sightMatrix[self.RInit][i] == 4):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit + 1, self.RInit * 2 + 1):
            if (sightMatrix[i][self.RInit] == 4):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)
        flag = 0
        for i in range(self.RInit + 1, self.RInit * 2 + 1):
            if (sightMatrix[self.RInit][i] == 4):
                self.output.append(abs(i - self.RInit))
                flag = 1
                break
        if (not flag):
            self.output.append(self.RInit)

        # DIRECTION
        ''' if(self.direction == pygame.K_DOWN):
                self.output.append(1)
            elif(self.direction == pygame.K_UP):
                self.output.append(2)
            elif(self.direction == pygame.K_LEFT):
                self.output.append(3)
            else:
                self.output.append(4)'''
        # SIZE
        # self.output.append(len(self.body))
        # SIGHT RADIUS
        # self.output.append(self.RInit)
        # SPEED
        # self.output.append(self.speed)

        # Food A*

        '''if self.foodPath:
            if self.foodPath[1][0] == self.RInit + 1 and self.direction != pygame.K_LEFT:
                self.output.append(-1)
            elif self.foodPath[1][0] == self.RInit - 1 and self.direction != pygame.K_RIGHT:
                self.output.append(1)
            else:
                self.output.append(0)

            if self.foodPath[1][1] == self.RInit + 1 and self.direction != pygame.K_UP:
                self.output.append(-1)
            elif self.foodPath[1][1] == self.RInit - 1 and self.direction != pygame.K_DOWN:
                self.output.append(1)
            else:
                self.output.append(0)
        else:
            self.output.append(0)
            self.output.append(0)'''

    def change_direction(self, key):
        if key == pygame.K_UP and self.direction != pygame.K_DOWN:
            self.change_to = pygame.K_UP
        if key == pygame.K_DOWN and self.direction != pygame.K_UP:
            self.change_to = pygame.K_DOWN
        if key == pygame.K_LEFT and self.direction != pygame.K_RIGHT:
            self.change_to = pygame.K_LEFT
        if key == pygame.K_RIGHT and self.direction != pygame.K_LEFT:
            self.change_to = pygame.K_RIGHT

    def move(self,map, action):
        self.fitness+=1
        if not self.playerControlled:
            '''if self.foodPath:
                if self.foodPath[1][0]==self.RInit+1 and self.direction != pygame.K_LEFT:
                    self.direction=pygame.K_RIGHT
                elif self.foodPath[1][0]==self.RInit-1 and self.direction != pygame.K_RIGHT:
                    self.direction=pygame.K_LEFT
                elif self.foodPath[1][1]==self.RInit+1 and self.direction != pygame.K_UP:
                    self.direction=pygame.K_DOWN
                elif self.foodPath[1][1]==self.RInit-1 and self.direction != pygame.K_DOWN:
                    self.direction=pygame.K_UP



            else:
                if self.direction==pygame.K_DOWN:
                    self.direction = random.choice([pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT])
                elif self.direction==pygame.K_UP:
                    self.direction = random.choice([pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT])
                elif self.direction==pygame.K_LEFT:
                    self.direction = random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT])
                else:
                    self.direction = random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_RIGHT])'''

            if (action == 0):
                #print("Keep direction")
                self.direction = self.direction
            if (action == 1):#LEFT
                #print("Move Left")
                if(self.direction==pygame.K_UP):
                    self.direction=pygame.K_LEFT
                elif (self.direction == pygame.K_LEFT):
                    self.direction = pygame.K_DOWN
                elif (self.direction == pygame.K_DOWN):
                    self.direction = pygame.K_RIGHT
                elif (self.direction == pygame.K_RIGHT):
                    self.direction = pygame.K_UP
            if (action == 2):#Right
                #print("Move right")
                if (self.direction == pygame.K_UP):
                    self.direction = pygame.K_RIGHT
                elif (self.direction == pygame.K_LEFT):
                    self.direction = pygame.K_UP
                elif (self.direction == pygame.K_DOWN):
                    self.direction = pygame.K_LEFT
                elif (self.direction == pygame.K_RIGHT):
                    self.direction = pygame.K_DOWN

        else:
            self.direction = self.change_to
        head_x, head_y = self.body[0]
        map[int(self.body[0][0] / 10)][int(self.body[0][1] / 10)] = 1
        if self.direction == pygame.K_UP:
            head_y -= 10
        if self.direction == pygame.K_DOWN:
            head_y += 10
        if self.direction == pygame.K_LEFT:
            head_x -= 10
        if self.direction == pygame.K_RIGHT:
            head_x += 10

        # Looping through the window
        #head_x = head_x % window_x
        #head_y = head_y % window_y
        #print (head_x)
        #print (window_x)
        if head_x < 0 or head_x >= window_x or head_y<0 or head_y>= window_y:
            #print("Mort")
            self.fitness-=1000
            print("WALL")
            return True

        new_head = [head_x, head_y]
        self.body.insert(0, new_head)

        if (map[int(self.body[0][0] / 10)][int(self.body[0][1] / 10)] == 1):
            self.fitness -= 1000
            print("Collision")
            return True
            return False
        map[int(self.body[0][0] / 10)][int(self.body[0][1] / 10)] = 2

        if not self.grow:
            x = self.body.pop()
            map[int(x[0] / 10)][int(x[1] / 10)] = 0

        else:
            self.grow = False

    def check_collision(self):
        # Will still probably have to do it like this, and iterate through every snake sadge
        return False
        # if map[int(self.body[0][0]/10)][int(self.body[0][1]/10)]==1:
        # return True

        '''for block in self.body[1:]:
            if self.body[0][0] == block[0] and self.body[0][1] == block[1]:
                return False
                #game_over()
                return True'''

    def grow_snake(self, map, grow_length):
        if (grow_length < 0):
            for i in range(grow_length * -1):
                x = self.body.pop()
                map[int(x[0] / 10)][int(x[1] / 10)] = 0
        else:
            for i in range(grow_length):
                self.body.insert(len(self.body) - 1, list(self.body[-1]))
                map[int(self.body[0][0] / 10)][int(self.body[0][1] / 10)] = 1
            self.grow = True

    def draw(self, game_window):
        for pos in self.body:
            # map[int(pos[0]/10)][int(pos[1]/10)]=1
            # print(int(pos[0]/10))
            pygame.draw.rect(game_window, self.color, pygame.Rect(pos[0], pos[1], 10, 10))


'''V_INIT = 55  # Should be between 50 and 60
R_INIT = 10
V_PAS = 2
R_Pas = 1
P = 500
PV = 500'''

#snakes = []
# snakes.append(Snake(green, [100, 100], 55,500,500,R_INIT,True))
# snakes.append(Snake(green, [150, 80], V_INIT,500,500,R_INIT,False))

# snake = Snake(green, [100, 50], V_INIT, 500, 500, R_INIT, True)
# snake2 = Snake(blue, [200, 50],True)

N_GEN = 5 * 60  # When a new food item is generated
#N_GENCounter = 0


class Food:
    expirationCounter = 0

    def __init__(self, NM, NExp, fruit_position, poisoned):
        self.NM = NM
        self.NExp = NExp * 60
        self.fruit_position = fruit_position
        self.poisoned = poisoned


M = 30  # Number of initial fruits
NM = 2  # Energy generated by food
N_EXP = 10  # After how much time it expires
'''fruits = []
for _ in range(M):
    fruits.append(
        Food(NM + random.randint(-1, 2), N_EXP + random.randint(-2, 2), [random.randrange(1, (window_x // 10)) * 10,
                                                                         random.randrange(1, (window_y // 10)) * 10],
             random.random() > 0.8))
for fruit in fruits:
    if fruit.poisoned:
        map[int(fruit.fruit_position[0] / 10)][int(fruit.fruit_position[1] / 10)] = 4
    else:
        map[int(fruit.fruit_position[0] / 10)][int(fruit.fruit_position[1] / 10)] = 3
'''

class Game:
    #snakes = []
    #map = [[0 for _ in range(48)] for _ in range(72)]
    #fruits = []
    #N_GENCounter=0
    def __init__(self,V_INIT,R_INIT,V_PAS,R_Pas,P,PV):
        self.snakes=[]
        self.map=[[0 for _ in range(48)] for _ in range(72)]
        self.fruits = []
        self.N_GENCounter=0
        self.V_INIT=V_INIT
        self.R_INIT=R_INIT
        self.V_PAS=V_PAS
        self.R_Pas=R_Pas
        self.P=P
        self.PV=PV
        for _ in range(M):
            self.fruits.append(
                Food(NM + random.randint(-1, 2), N_EXP + random.randint(-2, 2),
                     [random.randrange(1, (window_x // 10)) * 10,random.randrange(1,(window_y // 10)) * 10],random.random() > 0.8))
        #print(len(self.fruits))
        for fruit in self.fruits:
            if fruit.poisoned:
                self.map[int(fruit.fruit_position[0] / 10)][int(fruit.fruit_position[1] / 10)] = 4
            else:
                self.map[int(fruit.fruit_position[0] / 10)][int(fruit.fruit_position[1] / 10)] = 3
        self.snakes.append(Snake(green, [400,random.randrange(1,(window_y // 10)) * 10], V_INIT, P, PV, R_INIT, False))
        #self.snakes.append(Snake(green, [400,random.randrange(1,(window_y // 10)) * 10], V_INIT, P, PV, R_INIT, False))
        #self.snakes.append(Snake(green, [400, random.randrange(1, (window_y // 10)) * 10], V_INIT, P, PV, R_INIT, False))
        #self.snakes.append(Snake(green, [400, random.randrange(1, (window_y // 10)) * 10], V_INIT, P, PV, R_INIT, False))





    def tickSnake(self,index,action):
        global N_GEN,NM,N,N_EXP
        flag = False
        if self.snakes:

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    self.snakes[0].change_direction(event.key)



            try:
                if (len(self.snakes[index].body) < 3):
                    print("Starvation")
                    self.snakes.remove(self.snakes[index])
                    #print("Died of old age")
                    flag = True
                self.snakes[index].frameCounter = self.snakes[index].frameCounter + 1
                if (self.snakes[index].frameCounter == self.snakes[index].speed):
                    self.snakes[index].frameCounter = 0
                    # if (len(snake.body) >= 3):
                    if (self.snakes[index].move(self.map,action)):
                        # print("a")
                        flag = True
                        #print("Died of collision")
                        self.snakes.remove(self.snakes[index])



                self.snakes[index].pCounter = self.snakes[index].pCounter + 1
                self.snakes[index].pvCounter = self.snakes[index].pvCounter + 1

                if (self.snakes[index].pCounter == self.snakes[index].P):
                    x = self.snakes[index].body.pop()
                    self.map[int(x[0] / 10)][int(x[1] / 10)] = 0
                    self.snakes[index].pCounter = 0
                if (self.snakes[index].pvCounter >= self.snakes[index].PV):
                    self.snakes[index].speed = self.snakes[index].speed + self.V_PAS  # inversely proportional
                    self.snakes[index].RInit = self.snakes[index].RInit - self.R_Pas
                    self.snakes[index].pvCounter = 0

                for fruit in self.fruits:
                    #print (fruit.expirationCounter)
                    #print(fruit.NExp)
                    #print("")
                    fruit.expirationCounter = fruit.expirationCounter + 1
                    if self.snakes[index].body[0] == fruit.fruit_position:
                        #score += 10
                        self.snakes[index].fitness += 100
                        # fruit_spawn = False
                        if (fruit.poisoned):
                            if (len(self.snakes[index].body) - fruit.NM < 3):
                                self.snakes.remove(self.snakes[index])
                            else:
                                self.snakes[index].grow_snake(self.map,fruit.NM * -1)
                        else:
                            self.snakes[index].grow_snake(self.map,fruit.NM)
                        self.fruits.remove(fruit)

                    if (fruit.expirationCounter >= fruit.NExp):
                        self.fruits.remove(fruit)
                self.N_GENCounter = self.N_GENCounter + 1


                if (self.N_GENCounter == N_GEN):
                    self.N_GENCounter = 0
                    fruit_pos = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
                    self.fruits.append(
                        Food(NM + random.randint(-1, 2), N_EXP + random.randint(-2, 2), fruit_pos, random.random() > 0.8))
                    for fruit in self.fruits:
                        if fruit.poisoned:
                            self.map[int(fruit_pos[0] / 10)][int(fruit_pos[1] / 10)] = 4
                        else:
                            self.map[int(fruit_pos[0] / 10)][int(fruit_pos[1] / 10)] = 3

                ####A *
                sightMatrix = get_submatrix(self.map, (int(self.snakes[index].body[0][0] / 10), int(self.snakes[index].body[0][1] / 10)), self.snakes[index].RInit)
                sightMatrixRows = len(sightMatrix)
                sightMatrixCols = len(sightMatrix[0])


                # print("")
                goal = None
                poisonedGoal = None
                distance = 9999
                poisonedDistance = 9999
                '''start = (self.snakes[index].RInit, self.snakes[index].RInit)
                for r in range(sightMatrixRows):
                    for c in range(sightMatrixCols):
                        if (sightMatrix[r][c] == 3 and abs(r - self.R_INIT) + abs(c - self.R_INIT) < distance):
                            goal = (r, c)
                            distance = abs(r - self.R_INIT) + abs(c - self.R_INIT)
                        if (sightMatrix[r][c] == 4 and abs(r - self.R_INIT) + abs(c - self.R_INIT) < poisonedDistance):
                            poisonedGoal = (r, c)
                            poisonedDistance = abs(r - self.R_INIT) + abs(c - self.R_INIT)
                if (goal):
                    self.snakes[index].foodPath = Astar.a_star_search(sightMatrix, start, goal)
                else:
                    self.snakes[index].foodPath = []

                if (poisonedGoal):
                    self.snakes[index].poisonedFoodPath = Astar.a_star_search(sightMatrix, start, poisonedGoal)
                else:
                    self.snakes[index].poisonedFoodPath = []'''
                self.snakes[index].getOutput(sightMatrix)
                #print(index)
                #print(self.snakes[index].output)
            except:
                #print("aaa")
                pass

        if self.snakes:
            return False
        else:
            return True


'''game = Game(55,5,2,1,5000,500)
game2 = Game(55,10,2,1,500,500)
game3 = Game(55,10,2,1,500,500)
game4 = Game(55,10,2,1,500,500)
game5 = Game(55,10,2,1,500,500)
game6 = Game(55,10,2,1,500,500)
while game.snakes:

    for i,snake in enumerate(game.snakes):
        game.tickSnake(i,3)

    game_window.fill(black)
    for snake in game.snakes:
        snake.draw(game_window)
    # Draw other snakes if added

    for fruit in game.fruits:
        color = red if fruit.poisoned else white
        pygame.draw.rect(game_window, color, pygame.Rect(
            fruit.fruit_position[0], fruit.fruit_position[1], 10, 10))

    # displaying score continuously
    # show_score(1, white, 'times new roman', 20)

    # Refresh game screen
    pygame.display.update()

    # Frame Per Second /Refresh Rate

    fps.tick(60)

quit()'''



# initial score
score = 0


# displaying Score function
def show_score(choice, color, font, size):
    # creating font object score_font
    score_font = pygame.font.SysFont(font, size)

    # create the display surface object
    # score_surface
    score_surface = score_font.render('Score : ' + str(score), True, color)

    # create a rectangular object for the text
    # surface object
    score_rect = score_surface.get_rect()

    # displaying text
    game_window.blit(score_surface, score_rect)


# game over function
def game_over():
    # creating font object my_font
    my_font = pygame.font.SysFont('times new roman', 50)

    # creating a text surface on which text
    # will be drawn
    game_over_surface = my_font.render(
        'Your Score is : ' + str(score), True, red)

    # create a rectangular object for the text
    # surface object
    game_over_rect = game_over_surface.get_rect()

    # setting position of the text
    game_over_rect.midtop = (window_x / 2, window_y / 4)

    # blit will draw the text on screen
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()

    # after 2 seconds we will quit the program
    time.sleep(2)

    # play(score, snakes, fruits, map, window_x, window_y, game_window, fps,N_GENCounter)
    # deactivating pygame library
    # pygame.quit()

    # quit the program
    # quit()


gen = 0

import numpy as np



if __name__ == "__main__":
    EPISODES = 1000
    state_size = 12  # Example size, should match your state representation
    action_size = 3  # [straight, left, right]
    batch_size = 16

    agent = DQNAgent(state_size, action_size)
    game = Game(55,5,2,1,500,500)

    for e in range(EPISODES):
        game = Game(55, 5, 2, 1, 500, 500)
        state = game.snakes[0].output
        state = np.reshape(state, [1, state_size])
        for time in range(5000):
            fps.tick(60)
            action = agent.act(state)
            done = game.tickSnake(0, action)
            #print(action)
            try:
                print (game.snakes[0].output)
                next_state = game.snakes[0].output
                reward=game.snakes[0].fitness
            except:
                #print("FF")
                pass
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            game_window.fill(black)
            for snake in game.snakes:
                snake.draw(game_window)
            # Draw other snakes if added

            for fruit in game.fruits:
                color = red if fruit.poisoned else white
                pygame.draw.rect(game_window, color, pygame.Rect(
                    fruit.fruit_position[0], fruit.fruit_position[1], 10, 10))

            # displaying score continuously
            # show_score(1, white, 'times new roman', 20)

            # Refresh game screen
            pygame.display.update()


            if done:
                agent.update_target_model()
                print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save(f"snake-dqn-{e}.weights.h5")