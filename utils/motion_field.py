# Module to create a motion field given the depth map, rotation, and translation of the camera
# inputs and outputs are pytorch tensors
# Author: Hirak J Kashyap
import torch


@torch.no_grad()
def motion_mat(hh, ww, f):
    # Assume sensor dimensions are same as depth map
    # and camera center is in the center of the sensor
    N = hh * ww

    # Convert to unit focal length
    h = hh / f
    w = ww / f

    yv, xv = torch.meshgrid([torch.linspace(-h / 2, h / 2, hh), torch.linspace(-w / 2, w / 2, ww)])

    A = torch.zeros(2 * N, 3)
    B = torch.zeros(2 * N, 3)

    # xv_flat = xv.permute(1, 0).contiguous().view(-1, 1) #for MATLAB like column major
    # yv_flat = yv.permute(1, 0).contiguous().view(-1, 1)
    xv_flat = xv.contiguous().view(-1, 1)
    yv_flat = yv.contiguous().view(-1, 1)

    # MAY REQUIRE TO WITHDRAW THE UNIT FOCAL LENGTH ASSUMPTION

    A[range(0, 2 * N, 2), :] = torch.cat(
        (torch.ones(N, 1), torch.zeros(N, 1), -xv_flat), 1)
    A[range(1, 2 * N, 2), :] = torch.cat(
        (torch.zeros(N, 1), torch.ones(N, 1), -yv_flat), 1)

    B[range(0, 2 * N, 2), :] = torch.cat(
        (-xv_flat*yv_flat, 1 + xv_flat*xv_flat, -yv_flat), 1)

    B[range(1, 2 * N, 2), :] = torch.cat(
        (-1 - yv_flat * yv_flat, xv_flat*yv_flat, xv_flat), 1)

    return A.cuda(), B.cuda()


@torch.no_grad()
def motion_mat_fxfy(hh, ww, fx, fy):
    # Assume sensor dimensions are same as depth map
    # and camera center is in the center of the sensor
    N = hh * ww

    # Convert to unit focal length
    h = hh / fy
    w = ww / fx

    yv, xv = torch.meshgrid([torch.linspace(-h / 2, h / 2, hh), torch.linspace(-w / 2, w / 2, ww)])

    A = torch.zeros(2 * N, 3)
    B = torch.zeros(2 * N, 3)

    # xv_flat = xv.permute(1, 0).contiguous().view(-1, 1) #for MATLAB like column major
    # yv_flat = yv.permute(1, 0).contiguous().view(-1, 1)
    xv_flat = xv.contiguous().view(-1, 1)
    yv_flat = yv.contiguous().view(-1, 1)

    # MAY REQUIRE TO WITHDRAW THE UNIT FOCAL LENGTH ASSUMPTION

    A[range(0, 2 * N, 2), :] = torch.cat(
        (torch.ones(N, 1), torch.zeros(N, 1), -xv_flat), 1)
    A[range(1, 2 * N, 2), :] = torch.cat(
        (torch.zeros(N, 1), torch.ones(N, 1), -yv_flat), 1)

    B[range(0, 2 * N, 2), :] = torch.cat(
        (-xv_flat*yv_flat, 1 + xv_flat*xv_flat, -yv_flat), 1)

    B[range(1, 2 * N, 2), :] = torch.cat(
        (-1 - yv_flat * yv_flat, xv_flat*yv_flat, xv_flat), 1)
    return A.cuda(), B.cuda()


@torch.no_grad()
def gen_motion_field(t, w, depth, A, B):
    # Synthesize motion field given depth, camera motion(t, w), A, and B
    # t: 3X1 float tensor
    # w: 3X1 float tensor
    # A, B are obtained from motion_mat()
    hh = depth.size(0)
    ww = depth.size(1)
    #depth_flat = depth.permute(1, 0).contiguous().view(-1, 1) #for MATLAB like column major
    depth_flat = depth.contiguous().view(-1, 1)

    N = hh*ww
    inv_d = torch.zeros(2*N, 1).cuda()
    inv_d[range(0, 2 * N, 2), :] = 1/depth_flat
    inv_d[range(1, 2 * N, 2), :] = 1/depth_flat

    # v = torch.mm(inv_d * A, t) + torch.mm(B, w)
    # vx = v[range(0, 2 * N, 2)].view(hh, ww)
    # vy = v[range(1, 2 * N, 2)].view(hh, ww)
    # return vx, vy

    vt = torch.mm(inv_d * A, t)
    vw = torch.mm(B, w)
    vtx = vt[range(0, 2 * N, 2)].view(hh, ww)
    vty = vt[range(1, 2 * N, 2)].view(hh, ww)
    vwx = vw[range(0, 2 * N, 2)].view(hh, ww)
    vwy = vw[range(1, 2 * N, 2)].view(hh, ww)
    return vtx, vty, vwx, vwy


@torch.no_grad()
def est_translation(depth, A, vx, vy):
    # Least square estimate of ego translational velocity from depth and motion field,
    # when rotational velocity is zero
    hh = depth.size(0)
    ww = depth.size(1)
    N = hh * ww  # number of pixels

    # vectorize velocity observations (the order
    # here has to match the function motion_mat)
    vx_flat = vx.contiguous().view(-1, 1)
    vy_flat = vy.contiguous().view(-1, 1)
    v = torch.zeros(2 * N, 1).cuda()
    v[range(0, 2 * N, 2), :] = vx_flat
    v[range(1, 2 * N, 2), :] = vy_flat

    # vectorize inverse depth
    depth_flat = depth.contiguous().view(-1, 1)
    inv_d = torch.zeros(2*N, 1).cuda()
    inv_d[range(0, 2 * N, 2), :] = 1/depth_flat
    inv_d[range(1, 2 * N, 2), :] = 1/depth_flat

    # velocity field is given by:
    # v = rho*A*t + B*w
    # v = rho*A*t + 0
    # solve for unknown t (ego translation) as:
    # t = pinverse(rho*A)*v

    t_est = torch.mm(torch.pinverse(inv_d*A), v)
    return t_est


@torch.no_grad()
def est_rotation(B, vx, vy):
    # Least square estimate of ego rotational velocity from motion field,
    # when translational velocity is zero
    hh = vx.size(0)
    ww = vy.size(1)
    N = hh * ww  # number of pixels

    # vectorize velocity observations (the order
    # here has to match the function motion_mat)
    vx_flat = vx.contiguous().view(-1, 1)
    vy_flat = vy.contiguous().view(-1, 1)
    v = torch.zeros(2 * N, 1).cuda()
    v[range(0, 2 * N, 2), :] = vx_flat
    v[range(1, 2 * N, 2), :] = vy_flat

    # velocity field is given by:
    # v = rho*A*t + B*w
    # v = 0 + B*w
    # solve for unknown t (ego translation) as:
    # w = pinverse(B)*v
    w_est = torch.mm(torch.pinverse(B), v)
    return w_est


def motion_mat_f(h, w, f):
    # Assume sensor dimensions are same as depth map
    # and camera center is in the center of the sensor
    N = h * w

    # Convert to unit focal length
    # h = hh / f
    # w = ww / f

    yv, xv = torch.meshgrid([torch.linspace(-h / 2, h / 2, h), torch.linspace(-w / 2, w / 2, w)])

    A = torch.zeros(2 * N, 3)
    B = torch.zeros(2 * N, 3)

    # xv_flat = xv.permute(1, 0).contiguous().view(-1, 1) #for MATLAB like column major
    # yv_flat = yv.permute(1, 0).contiguous().view(-1, 1)
    xv_flat = xv.contiguous().view(-1, 1)
    yv_flat = yv.contiguous().view(-1, 1)

    # MAY REQUIRE TO WITHDRAW THE UNIT FOCAL LENGTH ASSUMPTION

    A[range(0, 2 * N, 2), :] = torch.cat(
        (f*torch.ones(N, 1), torch.zeros(N, 1), -xv_flat), 1)
    A[range(1, 2 * N, 2), :] = torch.cat(
        (torch.zeros(N, 1), f*torch.ones(N, 1), -yv_flat), 1)

    B[range(0, 2 * N, 2), :] = torch.cat(
        (-xv_flat*yv_flat, torch.ones_like(xv_flat)*f + xv_flat*xv_flat, -yv_flat), 1)

    B[range(1, 2 * N, 2), :] = torch.cat(
        (-torch.ones_like(yv_flat)*f - yv_flat * yv_flat, xv_flat*yv_flat, xv_flat), 1)

    return A.cuda(), B.cuda()


def gen_motion_field_f(t, w, depth, A, B, f):
    # Synthesize motion field given depth, camera motion(t, w), A, and B
    # t: 3X1 float tensor
    # w: 3X1 float tensor
    # A, B are obtained from motion_mat()
    hh = depth.size(0)
    ww = depth.size(1)
    #depth_flat = depth.permute(1, 0).contiguous().view(-1, 1) #for MATLAB like column major
    depth_flat = depth.contiguous().view(-1, 1)

    N = hh*ww
    inv_d = torch.zeros(2*N, 1).cuda()
    inv_d[range(0, 2 * N, 2), :] = f/depth_flat
    inv_d[range(1, 2 * N, 2), :] = f/depth_flat

    v = torch.mm(inv_d * A, t) + torch.mm(B, w)
    vx = v[range(0, 2 * N, 2)].view(hh, ww)
    vy = v[range(1, 2 * N, 2)].view(hh, ww)
    return vx, vy
