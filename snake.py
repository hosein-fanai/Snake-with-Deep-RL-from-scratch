from snake_body import SnakeBody

import numpy as np

import threading

import time

import msvcrt

import os


class Snake:

    def __init__(self, size):
        self.size = size

        self.eps = 1e-5
        self.frames_time = 0.4
        self.key = None
        self.snake = [SnakeBody()] # Snake's head

    def run(self):
        arr = np.zeros((self.size, self.size), dtype='uint8') # Or dtype=np.string
        arr[self.size//2, self.size//2] = 1
        arr = self.add_food(arr)

        direc = 'right' # Direction
        score = 0
        life = 1
        fps = 0
        self.snake[0].direction = direc
        self.snake[0].coords = (self.size//2, self.size//2)

        frames = 0
        start = time.time()
        while True:
            if life < 1:
                self.print_display(arr, direc, life, score, fps, mode='over')
                break
            
            self.print_display(arr, direc, life, score, fps)

            threading.Thread(target=self.get_input_key).start() # Threads need to be terminated
            time.sleep(self.frames_time)
            arr, direc, life, score = self.move_to_key(self.key, arr, direc, life, score)
            key = None

            frames += 1
            fps = int(frames // (time.time() - start + self.eps))

    def get_input_key(self):
        self.key = msvcrt.getch().decode().lower()

    def move_to_direc(self, arr, direc):
        arr = arr.copy()

        self.snake[0].direction = direc
        for k, snake_body in enumerate(self.snake):
            i, j = snake_body.coords

            match snake_body.direction:
                case 'left':
                    j -= 1
                case 'right':
                    j += 1
                case 'up':
                    i -= 1
                case 'down':
                    i += 1
            i, j = i%arr.shape[0], j%arr.shape[1]

            snake_body.coords = (i, j)
            
        arr = self.render_snake_on_display(arr)

        return arr, *self.snake[0].coords

    def move_to_key(self, key, arr, direc, life, score):
        match key:
            case 'a':
                if direc == 'right':
                    pass
                else:
                    direc = 'left'
            case 'd':
                if direc == 'left':
                    pass
                else:
                    direc = 'right'
            case 'w':
                if direc == 'down':
                    pass
                else:
                    direc = 'up'
            case 's':
                if direc == 'up':
                    pass
                else:
                    direc = 'down'
            case None:
                pass
            case _: # Do nothing
                return arr, direc, life, score 
            
        new_arr, head_i, head_j = self.move_to_direc(arr, direc)
        life, score, ate_food = self.check_collition(arr, life, score, head_i, head_j)
        self.backprop_direc()

        if ate_food:
            new_arr = self.add_food(new_arr)
            new_arr = self.add_to_body(new_arr)

        return new_arr, direc, life, score 

    def backprop_direc(self):
        prev_direc = self.snake[0].direction
        temp = None
        for snake_body in self.snake[0:]:
            temp = snake_body.direction
            snake_body.direction = prev_direc
            prev_direc = temp

        return

    def check_collition(self, arr, life, score, head_i, head_j):
        if arr[head_i, head_j] == 0:
            return life, score, False
        elif arr[head_i, head_j] in (2, 3):
            return life-1, score, False
        elif arr[head_i, head_j] == 4:
            return life, score+1, True # ate food

    def add_food(self, arr): # Could be written in a better way
        while True:
            a, b = np.random.randint(0, high=arr.shape[0], size=(2,))
            if arr[a, b] == 0:
                arr[a, b] = 4 # Food
                break
            
        return arr

    def add_to_body(self, arr):
        direc = self.snake[-1].direction
        i, j = self.snake[-1].coords

        match direc:
            case 'left':
                j+=1
            case 'right':
                j-=1
            case 'up':
                i+=1
            case 'down':
                i-=1

        self.snake.append(SnakeBody(direction=direc, coords=(i, j)))
        arr = self.render_snake_on_display(arr)

        return arr

    def render_snake_on_display(self, arr):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] in (1, 2, 3): # clear if is a snake_body
                    arr[i, j] = 0

        arr[self.snake[0].coords] = 1 # Head
        if len(self.snake) > 1:
            arr[self.snake[-1].coords] = 3 # Tale
        for snake_body in self.snake[1: -1]:
            arr[snake_body.coords] = 2

        return arr

    def print_display(self, arr, direc, life, score, fps, mode='on_going'):
        os.system('cls')
        print('life: ', life, 'score: ', score, 'fps: ', fps, sep='\t')

        if mode == 'on_going':
            print('-'*(arr.shape[0]+2))
            for i in range(arr.shape[0]):
                print('-', end='')
                for j in range(arr.shape[1]):
                    if arr[i, j] == 0: # Blank space
                        print(' ', end='')
                    elif arr[i, j] == 1: # Snake's head
                        snake = None
                        match direc:
                            case 'left':
                                snake = '<'
                            case 'right':
                                snake = '>'
                            case 'up':
                                snake = '^'
                            case 'down':
                                snake = 'v'
                        print(snake, end='')
                    elif arr[i, j] == 2: # Snake's body
                        print('=', end='')
                    elif arr[i, j] == 3: # Snake's tale
                        print(',', end='')
                    elif arr[i, j] == 4: # Food
                        print('F', end='')
                print('-', end='\n')
            print('-'*(arr.shape[0]+2))

        elif mode == 'over':
            print('\n\tGame over!')


if __name__ == "__main__":
    snake_game = Snake(size=10)
    snake_game.run()