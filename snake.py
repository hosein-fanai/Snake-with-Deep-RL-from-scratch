import pygame

from snake_body import SnakeBody

import numpy as np

import threading

import time

import msvcrt

import os


class Snake:
    epsilon = 1e-5
    frame_time = 0.4

    def __init__(self, size, pygame_mode=True):
        self.size = size
        self.multiplier = 30

        if pygame_mode:
            pygame.init()

            self.run = self.run2

            self.screen = pygame.display.set_mode(
                size=(self.size*self.multiplier, self.size*self.multiplier+50), 
                # flags=pygame.RESIZABLE
            )
            self.screen_rect = self.screen.get_rect()
            self.screen_dims = self.screen_rect.size

            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 50)

            pygame.display.set_caption("Snake")
        else:
            self.run = self.run1

        self.reset_game()

    def run1(self):
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

    def run2(self):
        while True:
            self._check_events()
            
            if self.life >= 1:
                self._move_to_key()
            
            self._update_screen()

            self.clock.tick(1/self.frame_time)

    def reset_game(self):
        self.arr = np.zeros((self.size, self.size), dtype='uint8') # Or dtype=np.string
        self.arr[self.size//2, self.size//2] = 1

        self._add_food()

        self.key = None
        self.direc = np.random.choice(['right', 'left', 'up', 'down']) # Direction
        self.score = 0
        self.life = 1

        self.snake = [SnakeBody()] # Snake's head
        self.snake[0].direction = self.direc
        self.snake[0].coords = (self.size//2, self.size//2)

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_a:
                        self.key = 'a'
                    case pygame.K_d:
                        self.key = 'd'
                    case pygame.K_w:
                        self.key = 'w'
                    case pygame.K_s:
                        self.key = 's'
                    case _:
                        self.key = None

    def _update_screen(self):
        self.screen.fill((200, 200, 200))

        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                cell_type = self.arr[i, j]
                if cell_type == 0:
                    continue
                elif cell_type == 1:
                    pygame.draw.rect(self.screen, (97, 2, 2), 
                                (j*self.multiplier, i*self.multiplier, self.multiplier, self.multiplier))
                elif cell_type == 2:
                    pygame.draw.rect(self.screen, (0, 0, 255), 
                                (j*self.multiplier, i*self.multiplier, self.multiplier, self.multiplier))
                elif cell_type == 3:
                    pygame.draw.rect(self.screen, (0, 255, 255), 
                                (j*self.multiplier, i*self.multiplier, self.multiplier, self.multiplier))
                elif cell_type == 4:
                    pygame.draw.rect(self.screen, (0, 255, 0), 
                                (j*self.multiplier, i*self.multiplier, self.multiplier, self.multiplier))

        for i in range(self.size+1):
            pygame.draw.line(self.screen, (0, 0, 0), (0, i*self.multiplier), (self.screen_dims[0], i*self.multiplier))
            pygame.draw.line(self.screen, (0, 0, 0), (i*self.multiplier, 0), (i*self.multiplier, self.screen_dims[0]))

        score_txt = self.font.render(f"Score: {self.score}", True, (0, 0, 0), (200, 200, 200))
        score_rect = score_txt.get_rect()
        score_rect.bottomleft = self.screen_rect.bottomleft
        self.screen.blit(score_txt, score_rect)

        if self.life < 1:
            text = self.font.render("Game Over", True, (0, 0, 0), (200, 200, 200))
            text_rect = text.get_rect()
            text_rect.center = self.screen_rect.center
            
            self.screen.blit(text, text_rect)

        # self._print_display(0)

        pygame.display.flip()

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

        prev_arr = self.arr.copy()

        head_i, head_j = self._move_to_direc()
        ate_food = self._check_collition(prev_arr, head_i, head_j)
        self._backprop_direc()

        if ate_food:
            self._add_to_body()
            self._add_food()

        self.key = None

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
        while np.where(self.arr == 0):
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