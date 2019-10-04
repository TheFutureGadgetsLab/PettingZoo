class Vector2():
    __slots__ = ['x', 'y']
    
    def __init__(self, x_iter_vec, y = None):
        if isinstance(x_iter_vec, Vector2):
            self.x = x_iter_vec.x
            self.y = x_iter_vec.y
        elif type(x_iter_vec) in [list, tuple]:
            self.x = x_iter_vec[0]
            self.y = x_iter_vec[1]
        else:
            self.x = x_iter_vec
            self.y = y

    def __eq__(self, other):
        if other == None:
            return False

        return (self.x == other.x) and (self.y == other.y)

    def __ne__(self, other):
        if other == None:
            return True

        return (self.x != other.x) or (self.y != other.y)

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __add__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)

        return Vector2(self.x + other, self.y + other)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)

        return Vector2(self.x - other, self.y - other)
    
    def __rsub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(other.x - self.x, other.y - self.y)
        
        return Vector2(other - self.x, other - self.y)

    def __mul__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x * other.x, self.y * other.y)

        return Vector2(self.x * other, self.y * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x / other.x, self.y / other.y)

        return Vector2(self.x / other, self.y / other)
    
    def __rdiv__(self, other):
        return self.__div__(other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x // other.x, self.y // other.y)

        return Vector2(self.x // other, self.y // other)
    
    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __iadd__(self, other):
        if isinstance(other, Vector2):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other
            self.y += other

        return self
    
    def __isub__(self, other):
        if isinstance(other, Vector2):
            self.x -= other.x
            self.y -= other.y
        else:
            self.x -= other
            self.y -= other

        return self
    
    def __imul__(self, other):
        if isinstance(other, Vector2):
            self.x *= other.x
            self.y *= other.y
        else:
            self.x *= other
            self.y *= other

        return self

    def __ifloordiv__(self, other):
        if isinstance(other, Vector2):
            self.x //= other.x
            self.y //= other.y
        else:
            self.x //= other
            self.y //= other
        
        return self
    
    def __idiv__(self, other):
        if isinstance(other, Vector2):
            self.x /= other.x
            self.y /= other.y
        else:
            self.x /= other
            self.y /= other

        return self
    
    def __int__(self):
        return Vector2(int(self.x), int(self.y))
    
    def __float__(self):
        return Vector2(float(self.x), float(self.y))
    
    def __abs__(self):
        return Vector2(abs(self.x), abs(self.y))
    
    def __round__(self):
        return Vector2(round(self.x), round(self.y))

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"({self.x}, {self.y})"