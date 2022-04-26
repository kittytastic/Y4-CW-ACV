import random

class Custom_AB_Loader:
    def __init__(self, A_loader, B_loader):
            self.A_size = len(A_loader)
            self.B_size = len(B_loader)

            self.A_loader = A_loader
            self.B_loader = B_loader

    def __getitem__(self, index):
        index_A = index%self.A_size
        index_B = random.randint(0, self.B_size - 1)
       
        A_data = self.A_loader[index_A]
        B_data = self.B_loader[index_B]

        merged_data = {**{f"A_{k}": v for k,v in A_data.items()}, **{f"B_{k}": v for k,v in B_data.items()}}

        return merged_data

    def __len__(self):
        return max(self.A_size, self.B_size)
