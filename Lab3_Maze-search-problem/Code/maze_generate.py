import random
import numpy as np


class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = np.ones((height, width), dtype=int)

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, pos):
        x, y = pos
        return [
            (x + 2, y),
            (x - 2, y),
            (x, y + 2),
            (x, y - 2)
        ]

    def generate_maze(self, start, extra_paths=20):
        stack = [start]
        self.maze[start] = 0

        while stack:
            x, y = current = stack[-1]
            self.maze[y, x] = 0

            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            valid_neighbors = []

            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if self.in_bounds((nx, ny)) and self.maze[ny, nx] == 1:
                    valid_neighbors.append((nx, ny))

            if valid_neighbors:
                next_pos = random.choice(valid_neighbors)
                stack.append(next_pos)
                self.maze[(y + next_pos[1]) // 2, (x + next_pos[0]) // 2] = 0
            else:
                stack.pop()

        for _ in range(extra_paths):
            while True:
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if self.maze[y, x] == 0:
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    for dx, dy in directions:
                        nx, ny = x + dx * 2, y + dy * 2
                        if self.in_bounds((nx, ny)) and self.maze[ny, nx] == 1:
                            self.maze[(y + ny) // 2, (x + nx) // 2] = 0
                            self.maze[ny, nx] = 0
                            break
                    break
        return self.maze

    def print_maze(self, start, end):
        x1, y1 = start
        x2, y2 = end
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if (x, y) == (x1, y1):
                    print('S', end='')
                elif (x, y) == (x2, y2):
                    print('E', end='')
                else:
                    print('0' if cell == 0 else '1', end='')
            print()


if __name__ == '__main__':
    height, width = 160, 240
    start = (1, 1)
    end = (width - 2, height - 2)
    maze_generator = Maze(width, height)
    maze = maze_generator.generate_maze(start, extra_paths=50)
    maze_generator.print_maze(start, end)
