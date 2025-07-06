import pygame
import random
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

class SnakeGame:
    def __init__(self, width=640, height=480):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head, 
                      Point(self.head.x - 10, self.head.y),
                      Point(self.head.x - 20, self.head.y)]
        self.direction = Direction.RIGHT
        self.food = None
        self._place_food()
        self.score = 0
        self.game_over = False

    def _place_food(self):
        x = random.randint(0, (self.width - 10) // 10) * 10
        y = random.randint(0, (self.height - 10) // 10) * 10
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 1. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # 2. check if game over
        reward = 0
        if self.is_collision():
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score

        # 3. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 4. update ui and clock
        self._update_ui()
        self.clock.tick(20)
        # 5. return game over and score
        return reward, self.game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.width - 10 or pt.x < 0 or \
           pt.y > self.height - 10 or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 10
        elif self.direction == Direction.LEFT:
            x -= 10
        elif self.direction == Direction.UP:
            y -= 10
        elif self.direction == Direction.DOWN:
            y += 10
        self.head = Point(x, y)

    def _update_ui(self):
        self.display.fill((0, 0, 0))

        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, 10, 10))

        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, 10, 10))

        text = pygame.font.Font(None, 25).render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

if __name__ == '__main__':
    game = SnakeGame()
    while not game.game_over:
        score, game_over = game.play_step()
    print('Final Score', score)
    pygame.quit()
