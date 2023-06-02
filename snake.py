from snake_body import SnakeBody

import numpy as np

import threading

import time

import msvcrt

import os


class Snake:
    epsilon = 1e-5
    frame_time = 0.4

    def __init__(self, size):
        self.size = size

        self.reset_game()

    def run(self):
        fps = 0
        frames = 0
        start = time.time()
        while True:
            if self.life < 1:
                self._print_display(fps, mode='over')
                break

            self._print_display(fps)

            threading.Thread(target=self._get_input_key).start() # Threads need to be terminated
            time.sleep(self.frame_time)

            self._move_to_key()
            self.key = None

            frames += 1
            fps = int(frames // (time.time() - start + self.epsilon))

    def reset_game(self):
        self.arr = np.zeros((self.size, self.size), dtype='uint8') # Or dtype=np.string
        self.arr[self.size//2, self.size//2] = 1

        self._add_food()

        self.key = None
        self.direc = 'right' # Direction
        self.score = 0
        self.life = 1

        self.snake = [SnakeBody()] # Snake's head
        self.snake[0].direction = self.direc
        self.snake[0].coords = (self.size//2, self.size//2)

    def _get_input_key(self):
        self.key = msvcrt.getch().decode().lower()

    def _move_to_direc(self):
        self.snake[0].direction = self.direc

        for snake_body in self.snake:
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
            i, j = i%self.arr.shape[0], j%self.arr.shape[1]

            snake_body.coords = (i, j)

        self._render_snake_on_display()

        return self.snake[0].coords

    def _move_to_key(self):
        match self.key:
            case 'a':
                if self.direc == 'right':
                    pass
                else:
                    self.direc = 'left'
            case 'd':
                if self.direc == 'left':
                    pass
                else:
                    self.direc = 'right'
            case 'w':
                if self.direc == 'down':
                    pass
                else:
                    self.direc = 'up'
            case 's':
                if self.direc == 'up':
                    pass
                else:
                    self.direc = 'down'
            case None:
                pass
            # case _: # Do nothing
            #     return

        prev_arr = self.arr.copy()

        head_i, head_j = self._move_to_direc()
        ate_food = self._check_collition(prev_arr, head_i, head_j)
        self._backprop_direc()

        if ate_food:
            self._add_food()
            self._add_to_body()

    def _backprop_direc(self):
        prev_direc = self.snake[0].direction
        temp = None
        for snake_body in self.snake[0:]:
            temp = snake_body.direction
            snake_body.direction = prev_direc
            prev_direc = temp

        return

    def _check_collition(self, prev_arr, head_i, head_j):
        if prev_arr[head_i, head_j] == 0:
            return False
        elif prev_arr[head_i, head_j] in (2, 3):
            self.life -= 1
            return False
        elif prev_arr[head_i, head_j] == 4:
            self.score += 1
            return True # ate food

    def _add_food(self): # Could be written in a better way
        while True:
            a, b = np.random.randint(0, high=self.size, size=(2,))
            if self.arr[a, b] == 0:
                self.arr[a, b] = 4 # Food
                break

    def _add_to_body(self):
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

        i, j = i%self.arr.shape[0], j%self.arr.shape[1]
        self.snake.append(SnakeBody(direction=direc, coords=(i, j)))

        self._render_snake_on_display()

    def _render_snake_on_display(self):
        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                if self.arr[i, j] in (1, 2, 3): # clear if is a snake_body
                    self.arr[i, j] = 0

        self.arr[self.snake[0].coords] = 1 # Head

        if len(self.snake) > 1:
            self.arr[self.snake[-1].coords] = 3 # Tail

        for snake_body in self.snake[1: -1]:
            self.arr[snake_body.coords] = 2 # Body

    def _print_display(self, fps, mode='on_going'):
        os.system('cls')
        print('life: ', self.life, 'score: ', self.score, 'fps: ', fps, sep='\t')

        if mode == 'on_going':
            print('-'*(self.arr.shape[0]+2))
            for i in range(self.arr.shape[0]):
                print('-', end='')
                for j in range(self.arr.shape[1]):
                    if self.arr[i, j] == 0: # Blank space
                        print(' ', end='')
                    elif self.arr[i, j] == 1: # Snake's head
                        snake = None
                        match self.direc:
                            case 'left':
                                snake = '<'
                            case 'right':
                                snake = '>'
                            case 'up':
                                snake = '^'
                            case 'down':
                                snake = 'v'
                        print(snake, end='')
                    elif self.arr[i, j] == 2: # Snake's body
                        print('=', end='')
                    elif self.arr[i, j] == 3: # Snake's tail
                        print(',', end='')
                    elif self.arr[i, j] == 4: # Food
                        print('F', end='')
                print('-', end='\n')
            print('-'*(self.arr.shape[0]+2))

        elif mode == 'over':
            print('\n\tGame over!')


if __name__ == "__main__":
    snake_game = Snake(size=10)
    snake_game.run()