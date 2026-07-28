import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { App } from "../app/App";
import { useDriverFlowStore } from "../stores/driverFlowStore";
import { useParkingStore } from "../stores/parkingStore";

describe("driver application flows", () => {
  beforeEach(() => {
    useParkingStore.getState().reset();
    useDriverFlowStore.getState().reset();
  });

  it("lets the driver skip recommendations and browse all statuses", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId("entry-skip"));
    expect(screen.getByTestId("browse-recommend")).toBeVisible();
    expect(screen.getByTestId("filter-all")).toHaveAttribute("aria-pressed", "true");
    expect(useDriverFlowStore.getState().mode).toBe("browse");
  });

  it("opens browse in empty-only mode from the entry sheet", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId("entry-empty"));
    expect(screen.getByTestId("filter-empty")).toHaveAttribute("aria-pressed", "true");
    const occupied = await screen.findByTestId("spot-A07");
    expect(occupied).toHaveAttribute("aria-disabled", "true");
  });

  it("requires explicit confirmation before a recommendation route is drawn", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId("entry-recommend"));
    await user.click(screen.getByTestId("need-shopping"));
    const result = await screen.findByTestId("recommendation-result");
    expect(result).toBeVisible();
    expect(screen.queryByTestId("active-route")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("recommendation-confirm"));
    expect(await screen.findByTestId("active-route")).toBeInTheDocument();
    expect(useDriverFlowStore.getState().mode).toBe("navigation");
  });

  it("allows manual navigation only from a green empty spot", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId("entry-skip"));
    await user.click(await screen.findByTestId("spot-A01"));
    expect(screen.getByRole("heading", { name: "Ô A01" })).toBeVisible();
    await user.click(screen.getByTestId("spot-navigate"));
    expect(await screen.findByTestId("active-route")).toBeInTheDocument();
  });

  it("allows inspecting but not navigating to a non-empty spot", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId("entry-skip"));
    await user.click(await screen.findByTestId("spot-A07"));
    expect(screen.getByRole("heading", { name: "Ô A07" })).toBeVisible();
    expect(screen.queryByTestId("spot-navigate")).not.toBeInTheDocument();
    expect(screen.getByText("Chỉ ô xanh đang trống mới có thể được chọn.")).toBeVisible();
  });

  it("pauses a confirmed route and shows the exact warning when status changes", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId("entry-skip"));
    await user.click(await screen.findByTestId("spot-A01"));
    await user.click(screen.getByTestId("spot-navigate"));
    expect(await screen.findByTestId("active-route")).toBeInTheDocument();

    act(() => {
      useParkingStore.getState().applyEvent({
        type: "spot.status.changed",
        cameraId: "cam-left",
        spotId: "A01",
        status: "transitioning",
        confidence: 0.93,
        revision: 2,
        updatedAt: "2026-07-25T08:05:00.000Z",
      });
    });

    await waitFor(() => expect(screen.getByRole("alertdialog")).toBeVisible());
    expect(screen.getByText("Ô A01 đang có phương tiện di chuyển vào hoặc ra.")).toBeVisible();
    expect(screen.queryByTestId("active-route")).not.toBeInTheDocument();
    expect(screen.getByTestId("switch-alternative")).toBeVisible();
  });
});
