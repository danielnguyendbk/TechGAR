import { expect, test, type Page } from "@playwright/test";

async function enterBrowse(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByTestId("entry-skip").click();
  await expect(page.getByTestId("browse-recommend")).toBeVisible();
}

async function enterRecommendation(page: Page, need: "shopping" | "services" | "entertainment"): Promise<void> {
  await page.goto("/");
  await page.getByTestId("entry-recommend").click();
  await page.getByTestId(`need-${need}`).click();
  await expect(page.getByTestId("recommendation-result")).toBeVisible();
}

test("entry to browse to manual empty-spot navigation", async ({ page }) => {
  await enterBrowse(page);
  await page.getByTestId("spot-A03").click();
  await expect(page.getByRole("heading", { name: "Ô A03" })).toBeVisible();
  await page.getByTestId("spot-navigate").click();
  await expect(page.getByTestId("active-route")).toBeVisible();
  await expect(page.getByText("Đang chỉ đường đến")).toBeVisible();
});

test("shopping recommendation does not route before confirmation", async ({ page }) => {
  await enterRecommendation(page, "shopping");
  await expect(page.getByTestId("active-route")).toHaveCount(0);
  await page.getByTestId("recommendation-confirm").click();
  await expect(page.getByTestId("active-route")).toBeVisible();
});

test("services recommendation can be abandoned for full browse", async ({ page }) => {
  await enterRecommendation(page, "services");
  await page.getByTestId("abandon-recommendation").click();
  await expect(page.getByTestId("browse-recommend")).toBeVisible();
  await expect(page.getByTestId("filter-all")).toHaveAttribute("aria-pressed", "true");
});

test("an unconfirmed entertainment recommendation recalculates when it turns amber", async ({ page }) => {
  await enterRecommendation(page, "entertainment");
  const bestSpot = page.getByTestId("recommendation-result").locator("strong").first();
  const originalId = await bestSpot.textContent();
  expect(originalId).toBeTruthy();

  await page.getByTestId("mock-toggle").click();
  await page.getByTestId("mock-scenario").selectOption("recommendation-transitioning");
  await page.getByTestId("mock-queue").click();
  await page.getByTestId("mock-step").click();

  await expect.poll(async () => bestSpot.textContent()).not.toBe(originalId);
  if (originalId) {
    await expect(page.getByTestId(`spot-${originalId}`)).toHaveAttribute("data-status", "transitioning");
  }
  await expect(page.getByTestId("active-route")).toHaveCount(0);
});

test("a confirmed spot becoming occupied pauses the route and offers an alternative", async ({ page }) => {
  await enterRecommendation(page, "shopping");
  const selectedId = await page.getByTestId("recommendation-result").locator("strong").first().textContent();
  await page.getByTestId("recommendation-confirm").click();
  await expect(page.getByTestId("active-route")).toBeVisible();

  await page.getByTestId("mock-toggle").click();
  await page.getByTestId("mock-scenario").selectOption("recommendation-occupied");
  await page.getByTestId("mock-queue").click();
  await page.getByTestId("mock-step").click();

  await expect(page.getByRole("alertdialog")).toBeVisible();
  if (selectedId) await expect(page.getByRole("alertdialog")).toContainText(`Ô ${selectedId} hiện không còn trống.`);
  await expect(page.getByTestId("active-route")).toHaveCount(0);
  await page.getByTestId("switch-alternative").click();
  await expect(page.getByTestId("active-route")).toBeVisible();
});

test("camera offline state is degraded without clearing owned spot status", async ({ page }) => {
  await enterBrowse(page);
  const statusBefore = await page.getByTestId("spot-A01").getAttribute("data-status");
  await page.getByTestId("mock-toggle").click();
  await page.getByTestId("mock-scenario").selectOption("cam-left-offline-recovery");
  await page.getByTestId("mock-queue").click();
  await page.getByTestId("mock-step").click();
  await expect(page.getByTestId("camera-health")).toContainText("1/2 online");
  await expect(page.getByTestId("spot-A01")).toHaveAttribute("data-status", statusBefore ?? "empty");
});

test("mobile map controls and desktop geometry remain available", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await enterBrowse(page);
  const svg = page.getByRole("img", { name: /Bản đồ 160 ô đỗ xe/ });
  const initialViewBox = await svg.getAttribute("viewBox");
  await page.getByLabel("Phóng to bản đồ").click();
  await expect.poll(async () => svg.getAttribute("viewBox")).not.toBe(initialViewBox);
  await page.getByTestId("reset-view").click();
  await expect(svg).toHaveAttribute("viewBox", "0 0 1200 900");

  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.locator("[data-spot-id]")).toHaveCount(160);
  await expect(page.getByTestId("spot-F10")).toBeVisible();
});

test("C10 renders one lane-valid route from the entrance through the C junction", async ({ page }) => {
  await enterBrowse(page);
  await page.getByTestId("spot-C10").click();
  await page.getByTestId("spot-navigate").click();

  const route = page.getByTestId("active-route");
  await expect(route).toBeVisible();
  await expect(route.locator("polyline.route-line")).toHaveCount(1);
  await expect(route.locator("polyline.route-line")).toHaveAttribute(
    "points",
    "997,858 997,437 621,437 621,422",
  );
  await expect(page.getByTestId("spot-C10")).toHaveAttribute("aria-current", "location");
});
