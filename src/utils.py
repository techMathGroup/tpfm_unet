import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def mask_lambda(x, y, lambda_index=0):
    mask = 1.0 - y[:, lambda_index, :, :]
    x[:, :3, :, :] *= mask.unsqueeze(1)
    return x


def mask_flow(x, y, flow_index=4):
    mask = y[:, flow_index, :, :]
    x[:, :3, :, :] *= mask.unsqueeze(1)
    return x


def pad_to_even(x):
    h, w = x.shape[-2:]
    pad_h = (h % 2)
    pad_w = (w % 2)
    return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')


def center_crop(enc_feat, target_feat):
    _, _, h, w = target_feat.shape
    # print("Target shape:", h, w)
    enc_h, enc_w = enc_feat.shape[-2:]
    # print("Enc shape:", enc_h, enc_w)
    assert enc_h >= h
    assert enc_w >= w
    crop_top = (enc_h - h) // 2
    crop_left = (enc_w - w) // 2
    return enc_feat[:, :, crop_top:crop_top + h, crop_left:crop_left + w]


def compute_y_stats(data_points):
    # Stack all y's (first 4 channels) along new batch dim
    y_all = torch.stack([dp[:4] for dp in data_points], dim=0)  # shape: [N, 4, H, W]

    mean = y_all.mean(dim=[0, 2, 3])  # mean per channel (4,)
    std = y_all.std(dim=[0, 2, 3])  # std per channel (4,)

    return mean, std


def normalize_y(sample, mean, std, eps=1e-6):
    return (sample - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + eps)


def denormalize_y(sample, mean, std, eps=1e-6):
    return sample * (std.view(-1, 1, 1) + eps) + mean.view(-1, 1, 1)


def plot_quiver(data_point, coords_array, padding=0.01, quiver_scale=1e2, ax=None):
    if data_point.shape[0] == 4:
        Ux, Uy, Uz, p = data_point
    elif data_point.shape[0] == 3:
        Ux, Uy, Uz = data_point
    else:
        print(f"Invalid data for plotting quiver. shape[0] = {data_point.shape[0]}")
        return

    # Extract coordinates and reshape to match grid
    x_dim = data_point.shape[1]
    y_dim = data_point.shape[2]
    # print(f"x_dim: {x_dim}, y_dim: {y_dim}")
    x = coords_array[:, 0].reshape(x_dim, y_dim)
    y = coords_array[:, 1].reshape(x_dim, y_dim)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    try:
        ax.quiver(x, y, Ux.numpy(), Uy.numpy(), scale=quiver_scale, scale_units='xy', color='blue')
    except AttributeError:
        ax.quiver(x, y, Ux, Uy, scale=quiver_scale, scale_units='xy', color='blue')
    ax.set_title('Velocity Field (Ux, Uy) with Real Coordinates')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.grid()
    if ax is None:
        plt.show()


def plot_velocity_magnitude(data_point, coords_array, ax=None, extent=None, vmin=0, vmax=None):
    if data_point.shape[0] == 4:
        Ux, Uy, Uz, p = data_point
    elif data_point.shape[0] == 3:
        Ux, Uy, Uz = data_point
    else:
        print("Invalid data for plotting velocity magnitude: shape[0] =", data_point.shape[0])
        return

    # Compute velocity magnitude
    velocity_magnitude = torch.sqrt(Ux ** 2 + Uy ** 2 + Uz ** 2)

    # Extract coordinates and reshape to match grid
    x_dim = data_point.shape[1]
    y_dim = data_point.shape[2]
    x = coords_array[:, 0].reshape(x_dim, y_dim)
    y = coords_array[:, 1].reshape(x_dim, y_dim)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(velocity_magnitude.detach().numpy() if hasattr(velocity_magnitude, "detach") else velocity_magnitude,
              extent=(extent if extent else (x.min(), x.max(), y.min(), y.max())),
              origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title('Velocity Magnitude')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(ax.images[0], ax=ax, label='Velocity Magnitude')
    if ax is None:
        plt.show()


def display_velocity_magnitude(data_point, coords_array, ax=None, extent=None, vmin=0, vmax=None):
    # --- Extract fields ---
    if data_point.shape[0] == 4:
        Ux, Uy, Uz, p = data_point
    elif data_point.shape[0] == 3:
        Ux, Uy, Uz = data_point
    else:
        raise ValueError("Invalid data for plotting velocity magnitude")

    # --- Compute velocity magnitude ---
    velocity_magnitude = torch.sqrt(Ux**2 + Uy**2 + Uz**2)

    # --- Coordinates ---
    x_dim = data_point.shape[1]
    y_dim = data_point.shape[2]
    x = coords_array[:, 0].reshape(x_dim, y_dim)
    y = coords_array[:, 1].reshape(x_dim, y_dim)

    # --- Prepare figure ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_fig = True
    else:
        fig = ax.figure

    # --- Convert magnitude to numpy ---
    data = velocity_magnitude.detach().cpu().numpy()

    # --- Plot ---
    img = ax.imshow(
        data,
        extent=(extent if extent else (x.min(), x.max(), y.min(), y.max())),
        origin='lower',
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(img, ax=ax, label='Velocity Magnitude')

    ax.set_title("Velocity Magnitude")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # ============================================================
    # =============  CLICK-TO-INSPECT TOOLTIP  ===================
    # ============================================================

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white")
    )
    annot.set_visible(False)

    def onclick(event):
        if event.inaxes != ax:
            return

        # Convert click coordinates (event.xdata, event.ydata) → array indices
        ix = int((event.xdata - x.min()) / (x.max() - x.min()) * (x_dim - 1))
        iy = int((event.ydata - y.min()) / (y.max() - y.min()) * (y_dim - 1))

        if 0 <= ix < x_dim and 0 <= iy < y_dim:
            val = data[iy, ix]

            # Position the tooltip
            annot.xy = (event.xdata, event.ydata)
            annot.set_text(f"{val:.4f}")
            annot.set_visible(True)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)

    if created_fig:
        plt.show()

    return ax


def plot_pressure(data_point, coords_array, ax=None):
    if data_point.shape[0] == 4:
        Ux, Uy, Uz, p = data_point
    elif data_point.shape[0] == 1:
        p = data_point[0]
    else:
        print("Invalid data for plotting pressure: shape[0] =", data_point.shape[0])
        return

    # Extract coordinates and reshape to match 20x20 grid
    x_dim = data_point.shape[1]
    y_dim = data_point.shape[2]
    x = coords_array[:, 0].reshape(x_dim, y_dim)
    y = coords_array[:, 1].reshape(x_dim, y_dim)

    # Surface plot for pressure
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True
    ax.plot_surface(x, y, p.detach().numpy() if hasattr(p, "detach") else p, cmap='viridis')
    ax.set_title('Pressure Field (p)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Pressure')
    if created_fig:
        plt.show()


def load_coordinates(coords_file):
    coords_df = pd.read_csv(coords_file)
    coords_array = coords_df.drop(columns=['cellI']).to_numpy()
    return coords_array


def log_predictions(x, y, output, mlf_logger):
    x = x.cpu()
    y = y.cpu()
    output = output.cpu()

    # Load coordinates
    # coords_array = x[0, -3:, :, :].reshape(3, -1).T.numpy()
    size = x.shape[2]
    assert size in [64, 128, 256], "Unsupported size for loading coordinates."
    assert x.shape[2] == x.shape[3], "Input must be square."
    coords_array = load_coordinates(f'./data/coordinates_{size}.csv')

    for batch_index in tqdm(range(x.shape[0]), desc="Visualizing samples"):
        # Outputs
        fig, axes = plt.subplots(2, 2, figsize=(15, 14))
        # plot_quiver(output[batch_index], coords_array, quiver_scale=50, ax=axes[0][0])
        # plot_quiver(y[batch_index], coords_array, quiver_scale=50, ax=axes[0][1])
        Ux, Uy, Uz, _ = output[batch_index]
        velocity_magnitude1 = torch.sqrt(Ux ** 2 + Uy ** 2 + Uz ** 2)
        Ux_gt, Uy_gt, Uz_gt, _ = y[batch_index]
        velocity_magnitude2 = torch.sqrt(Ux_gt ** 2 + Uy_gt ** 2 + Uz_gt ** 2)
        vmax = max(velocity_magnitude1.max().item(), velocity_magnitude2.max().item())

        plot_velocity_magnitude(output[batch_index], coords_array, ax=axes[0][0], vmin=0, vmax=vmax)
        plot_velocity_magnitude(y[batch_index], coords_array, ax=axes[0][1], vmin=0, vmax=vmax)

        # Bottom row: replace with 3D axes for pressure
        fig.delaxes(axes[1, 0])  # remove 2D axis
        fig.delaxes(axes[1, 1])
        axes[1, 0] = fig.add_subplot(2, 2, 3, projection='3d')
        axes[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')

        plot_pressure(output[batch_index], coords_array, ax=axes[1][0])  # needs ax support
        plot_pressure(y[batch_index], coords_array, ax=axes[1][1])  # needs ax support

        fig.text(0.25, 0.95, "Model Output", ha="center", fontsize=16)
        fig.text(0.75, 0.95, "Ground Truth (OpenFOAM)", ha="center", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        mlf_logger.experiment.log_figure(
            mlf_logger.run_id, fig, f"predictions/{batch_index}/compare_velocity_pressure.png")
        plt.close(fig)

        # Input
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        lambda_img = x[batch_index, 0, :, :]
        cax = ax2.imshow(lambda_img, cmap='gray', origin='lower')
        ax2.set_title('Input Topology (Lambda)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        fig2.colorbar(cax, ax=ax2, label='Lambda Value')
        plt.tight_layout()
        mlf_logger.experiment.log_figure(
            mlf_logger.run_id, fig2, f"predictions/{batch_index}/input_topology.png")
        plt.close(fig2)
