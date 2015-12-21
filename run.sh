set -x
num_procs=8
#num_procs=32
probSize=13378200 # MB contributed to allreduce by each node
# probSize=26756400 # 200%
# probSize=1337820 # 10%

executable=./main

# hostfile=hostfiles/f1_f2_1slot.txt
hostfile=hostfiles/firebox_1slot.txt
# cat $hostfile

mpirun=/opt/openmpi/bin/mpirun

# $mpirun --hostfile $hostfile -np $num_procs $executable $probSize
# $mpirun --mca btl_openib_if_include "mlx4_1:1" --mca btl self,openib --mca pml cm --mca mtl mxm  --hostfile $hostfile -np $num_procs $executable $probSize
$mpirun --mca btl_openib_if_include "mlx4_1:1" --mca btl self,openib --mca pml ob1 --hostfile $hostfile -np $num_procs $executable $probSize
